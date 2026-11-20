from dm_control import manipulation, suite
import copy
import json
import math
import os
import random
import shutil
import sys
import time

from matplotlib.pyplot import step

import env as ENV

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from logger import Logger
from omegaconf import DictConfig, open_dict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from mjpr import mjpr

import wandb

CLI_ENV_NAME = None
CLI_SEED = None


def _extract_cli_args(argv):
    """Extract short CLI arguments before Hydra parses them."""
    cleaned = [argv[0]]
    env_name = None
    seed = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-env", "--env"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for -env, for example: -env walker/walk")
            env_name = argv[i + 1]
            i += 2
        elif arg.startswith(("-env=", "--env=")):
            env_name = arg.split("=", 1)[1]
            i += 1
        elif arg in ("-seed", "--seed"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for -seed, for example: -seed 1111")
            seed = int(argv[i + 1])
            i += 2
        elif arg.startswith(("-seed=", "--seed=")):
            seed = int(arg.split("=", 1)[1])
            i += 1
        else:
            cleaned.append(arg)
            i += 1
    return cleaned, env_name, seed


def _append_hydra_run_dir(argv, env_name, seed):
    """Route Hydra output into outputs/date/task/time-seed."""
    if not env_name:
        return argv
    if any(arg.startswith("hydra.run.dir=") for arg in argv[1:]):
        return argv
    task_name = env_name.split("/", 1)[1] if "/" in env_name else env_name
    safe_task_name = task_name.replace("/", "_").replace("\\", "_")
    seed_label = f"seed{seed}" if seed is not None else "seed${seed_and_gpuid.0}"
    run_dir = (
        f"./outputs/${{now:%Y-%m-%d}}/{safe_task_name}/"
        f"${{now:%H-%M-%S}}-{seed_label}"
    )
    return argv + [f"hydra.run.dir={run_dir}"]


def _resolve_env_config(args, env_name):
    if not env_name:
        raise ValueError(
            "Please pass -env, for example: python multi-modal_train.py -env walker/walk"
        )
    if "/" not in env_name:
        raise ValueError(f"env_name must use domain/task format, got: {env_name}")
    if env_name not in args.envs:
        available = ", ".join(args.envs.keys())
        raise KeyError(f"Unknown env_name '{env_name}'. Available envs: {available}")

    env_cfg = args.envs[env_name]
    if "action_shape" not in env_cfg or "num_env_steps" not in env_cfg:
        raise KeyError(
            f"envs.{env_name} must define action_shape and num_env_steps."
        )
    domain_name, task_name = env_name.split("/", 1)
    with open_dict(args):
        args.env_name = env_name
        args.domain_name = domain_name
        args.task_name = task_name
        args.action_shape = int(env_cfg.action_shape)
        args.num_env_steps = int(env_cfg.num_env_steps)
    return env_name, domain_name, task_name


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 batch_size,
                 device,
                 image_size=84,
                 transform=None,
                 auxiliary_task_batch_size=64,
                 jumps=5):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.jumps = jumps
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32

        self.image_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.depth_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_image_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_depth_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32)  # for auxiliary task

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.current_auxiliary_batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.image_obses[self.idx], obs[0])
        np.copyto(self.depth_obses[self.idx], obs[1])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_image_obses[self.idx], next_obs[0])
        np.copyto(self.next_depth_obses[self.idx], next_obs[1])
        np.copyto(self.not_dones[self.idx], not done)  # "not done" is always True
        np.copyto(self.real_dones[self.idx], isinstance(done, int))  # "not done" is always True
        # print(not done, isinstance(done, int))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        image_obses = self.image_obses[idxs]
        depth_obses = self.depth_obses[idxs]
        next_image_obses = self.next_image_obses[idxs]
        next_depth_obses = self.next_depth_obses[idxs]

        image_obses = torch.as_tensor(image_obses, device=self.device).float()
        depth_obses = torch.as_tensor(depth_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_image_obses = torch.as_tensor(next_image_obses, device=self.device).float()
        next_depth_obses = torch.as_tensor(next_depth_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses,depth_obses], actions, rewards, [next_image_obses,next_depth_obses], not_dones

    # v2
    def sample_spr(self):  # sample batch for auxiliary task
        idxs = np.random.randint(0,
                                 self.capacity - self.jumps -
                                 1 if self.full else self.idx - self.jumps - 1,
                                 size=self.auxiliary_task_batch_size * 2)
        #  size=self.auxiliary_task_batch_size)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.jumps + 1).reshape(1, -1)  # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)  # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1) == 0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs]  # (B, jumps+1)
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        depth_obses = torch.as_tensor(self.depth_obses[idxs], device=self.device).float()
        #next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        #next_depth_obses = torch.as_tensor(self.next_depth_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        #not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)  # (B, jumps+1, 1)

        spr_samples = {
            'image': image_obses.transpose(0, 1).unsqueeze(3),
            'depth': depth_obses.transpose(0, 1).unsqueeze(3),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        return (*self.sample(), spr_samples)

    def sample(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        depth_obses = torch.as_tensor(self.depth_obses[idxs], device=self.device).float()
        next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        next_depth_obses = torch.as_tensor(self.next_depth_obses[idxs], device=self.device).float()


        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses,depth_obses], actions, rewards,[next_image_obses,next_depth_obses], not_dones



    def add_agent(self, agent):
        if hasattr(agent, 'SPR'):
            self.SPR = agent.SPR




class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False



def evaluate(env, agent, num_episodes, L, step):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()

            done = False
            episode_reward = 0
            while not done:
                # center crop image

                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)

                episode_reward += reward


            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        print('eval:',mean_ep_reward,best_ep_reward)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
        return mean_ep_reward

    return run_eval_loop(sample_stochastically=False)
     


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'mjpr':
        return mjpr(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            model_spr_weight=args.fp_loss_weight,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim)



@hydra.main(config_path=".", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    # 启用梯度异常检测（调试用）
    torch.autograd.set_detect_anomaly(True)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    env_name, domain_name, task_name = _resolve_env_config(args, CLI_ENV_NAME)
    env_id = env_name.replace('/', '_')
    if CLI_SEED is not None:
        with open_dict(args):
            args.seed = int(CLI_SEED)

    wandb.init(
        project="MJPR",
        mode = "disabled",
        name=args.agent + '_' + env_id + '_' + str(args.seed)
    )
    reward_all=[]
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    torch.cuda.set_device(args.gpuid)
    set_seed_everywhere(args.seed)
    seed=int(args.seed)
    env = suite.load(domain_name, task_name, task_kwargs={'random': seed}, visualize_reward=False)
    env = ENV.make(env, action_repeat=args.action_repeat, mode='mm')



    os.makedirs(output_dir, exist_ok=True)
    args.work_dir = output_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.join(script_dir, 'mjpr.py'), os.path.join(output_dir, 'mjpr.py'))
    shutil.copy2(__file__, os.path.join(output_dir, os.path.basename(__file__)))
    shutil.copy2(os.path.join(script_dir, 'config.yaml'), os.path.join(output_dir, 'config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = [int(args.action_shape)]

    if args.encoder_type == 'pixel':
        obs_shape = (args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (args.frame_stack,
                             args.pre_transform_image_size,
                             args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        auxiliary_task_batch_size=args.auxiliary_task_batch_size,
        jumps=args.jumps,
    )

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)
    replay_buffer.add_agent(agent)

    L = Logger(args.work_dir, use_tb=args.save_tb, use_wandb=args.wandb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    frame = [np.zeros((84, 84)), np.zeros((84, 84)), np.zeros((84, 84))]
    for step in range(0, args.num_env_steps, args.action_repeat):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            r=evaluate(env, agent, args.num_eval_episodes, L, step)
            reward_all.append(r)
            reward_path = os.path.join(
                output_dir,
                f'reward_{env_id}_{args.agent}_{args.seed}.npy',
            )
            np.save(reward_path, np.array(reward_all))


        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
                L.log('train/episode_reward', episode_reward, step)
            
            print('episode_reward', episode_reward, step)
            wandb.log(data={'reward':episode_reward}, step=step)
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        if step == 300000:
            agent.save(
                env_id,
                args.agent,
                str(args.seed),
                step,
                model_dir=output_dir,
            )
            agent.save_spr(
                env_id,
                args.agent,
                str(args.seed),
                step,
                model_dir=output_dir,
            )



        # sample action for data collection
        if step < args.init_steps:
            action = np.random.rand(args.action_shape)*2-1
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap

        done_bool = 0 if episode_step + 1 == 250 else float(
            done)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        obs=next_obs

        episode_step += 1

    final_step = args.num_env_steps
    agent.save_spr(
        env_id,
        args.agent,
        str(args.seed),
        final_step,
        model_dir=output_dir,
    )
    wandb.finish()


if __name__ == '__main__':
    sys.argv, CLI_ENV_NAME, CLI_SEED = _extract_cli_args(sys.argv)
    sys.argv = _append_hydra_run_dir(sys.argv, CLI_ENV_NAME, CLI_SEED)
    torch.multiprocessing.set_start_method('spawn')
    main()
