import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

from dm_control import manipulation, suite
import copy
import json
import math
import random
import sys
import time

import env as ENV

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from logger import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from mjpr import mjpr

import wandb

CLI_ENV_NAME = None
CLI_SEED = None


def _extract_cli_args(argv):
    """Extract short CLI arguments before Hydra parses command line arguments."""
    cleaned = [argv[0]]
    env_name = None
    seed = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-env", "--env"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for -env. Example: python multi-modal_train.py -env walker/walk")
            env_name = argv[i + 1]
            i += 2
        elif arg.startswith("-env="):
            env_name = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--env="):
            env_name = arg.split("=", 1)[1]
            i += 1
        elif arg in ("-seed", "--seed"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for -seed. Example: python multi-modal_train.py -env walker/walk -seed 10")
            seed = int(argv[i + 1])
            i += 2
        elif arg.startswith("-seed="):
            seed = int(arg.split("=", 1)[1])
            i += 1
        elif arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])
            i += 1
        else:
            cleaned.append(arg)
            i += 1
    return cleaned, env_name, seed


def _resolve_run_seed(argv, cli_seed):
    """Resolve the seed used in the Hydra output directory."""
    if cli_seed is not None:
        return int(cli_seed)
    for arg in argv[1:]:
        if arg.startswith("seed="):
            return int(arg.split("=", 1)[1])

    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    configured_seed = int(config.get("seed", 0) or 0)
    if configured_seed != 0:
        return configured_seed
    return int(config.seed_and_gpuid[0])


def _append_hydra_run_dir(argv, env_name, seed):
    """Route Hydra output into outputs/date/task/time-seed."""
    if not env_name:
        return argv
    if any(arg.startswith("hydra.run.dir=") for arg in argv[1:]):
        return argv
    task_name = env_name.split("/", 1)[1] if "/" in env_name else env_name
    safe_task_name = task_name.replace("/", "_").replace("\\", "_")
    run_dir = (
        f"./outputs/${{now:%Y-%m-%d}}/{safe_task_name}/"
        f"${{now:%H-%M-%S}}-seed{seed}"
    )
    return argv + [f"hydra.run.dir={run_dir}"]


def _resolve_env_config(args, env_name):
    if not env_name:
        raise ValueError("Please pass the environment with -env, for example: python multi-modal_train.py -env walker/walk")
    if "/" not in env_name:
        raise ValueError(f"env_name must use 'domain/task' format, got: {env_name}")
    if "envs" not in args:
        raise KeyError("config.yaml must define an envs mapping with action_shape and num_env_steps for each env_name.")
    if env_name not in args.envs:
        available = ", ".join(args.envs.keys())
        raise KeyError(f"Unknown env_name '{env_name}'. Available envs: {available}")

    env_cfg = args.envs[env_name]
    if "action_shape" not in env_cfg or "num_env_steps" not in env_cfg:
        raise KeyError(f"envs.{env_name} must define action_shape and num_env_steps.")

    domain_name, task_name = env_name.split("/", 1)
    with open_dict(args):
        args.env_name = env_name
        args.action_shape = int(env_cfg.action_shape)
        args.num_env_steps = int(env_cfg.num_env_steps)
    return env_name, domain_name, task_name


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        assert len(obs_shape) == 2
        image_obs_shape = obs_shape[0]
        position_obs_shape = obs_shape[1]
        self.image_obses = np.empty((capacity, *image_obs_shape), dtype=obs_dtype)
        self.position_obses = np.empty((capacity, *position_obs_shape), dtype=obs_dtype)
        self.next_image_obses = np.empty((capacity, *image_obs_shape), dtype=obs_dtype)
        self.next_position_obses = np.empty((capacity, *position_obs_shape), dtype=obs_dtype)
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
        np.copyto(self.position_obses[self.idx], obs[1])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_image_obses[self.idx], next_obs[0])
        np.copyto(self.next_position_obses[self.idx], next_obs[1])
        np.copyto(self.not_dones[self.idx], not done)  # "not done" is always True
        # np.copyto(self.real_dones[self.idx], isinstance(done, int))  # "not done" is always True
        np.copyto(self.real_dones[self.idx], bool(done)) 
        # print(not done, isinstance(done, int))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        image_obses = self.image_obses[idxs]
        position_obses = self.position_obses[idxs]
        next_image_obses = self.next_image_obses[idxs]
        next_position_obses = self.next_position_obses[idxs]

        image_obses = torch.as_tensor(image_obses, device=self.device).float()
        position_obses = torch.as_tensor(position_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_image_obses = torch.as_tensor(next_image_obses, device=self.device).float()
        next_position_obses = torch.as_tensor(next_position_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses, position_obses], actions, rewards, [next_image_obses, next_position_obses], not_dones

    # v2
    def sample_spr(self):  # sample batch for auxiliary task
        upper = self.capacity - self.jumps - 1 if self.full else self.idx - self.jumps - 1
        if upper <= 0:
            raise ValueError(
                f"Not enough replay data for sample_spr: need horizon {self.jumps}, current idx {self.idx}"
            )

        base_idxs = np.random.randint(0, upper, size=self.auxiliary_task_batch_size * 2)
        base_idxs = base_idxs.reshape(-1, 1)
        idxs = base_idxs + np.arange(self.jumps + 1).reshape(1, -1)

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)  # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_rows = torch.where((real_dones.mean(1) == 0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_rows]
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        action_idxs = idxs

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        position_obses = torch.as_tensor(self.position_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[action_idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[action_idxs], device=self.device)

        spr_samples = {
            'image': image_obses.transpose(0, 1).unsqueeze(3),
            'position': position_obses.transpose(0, 1),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        return (*self.sample(), spr_samples)

    def sample(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        position_obses = torch.as_tensor(self.position_obses[idxs], device=self.device).float()
        next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        next_position_obses = torch.as_tensor(self.next_position_obses[idxs], device=self.device).float()


        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses, position_obses], actions, rewards, [next_image_obses, next_position_obses], not_dones

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
        print(f'eval: mean={mean_ep_reward:.2f}, best={best_ep_reward:.2f}')
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
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
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
            detach_encoder=args.detach_encoder)
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    env_name, domain_name, task_name = _resolve_env_config(args, CLI_ENV_NAME)
    env_id = env_name.replace('/', '_')
    
    wandb_mode = "online" if args.wandb else "disabled"
    wandb.init(
        project="MJPR",
        mode = wandb_mode,
        name=args.agent + '_' + env_id + '_' + str(args.seed)
    )
    reward_all=[]
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    if CLI_SEED is not None:
        with open_dict(args):
            args.seed = int(CLI_SEED)
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    torch.cuda.set_device(args.gpuid)
    set_seed_everywhere(args.seed)
    seed=int(args.seed)
    env = suite.load(domain_name, task_name, task_kwargs={'random': seed}, visualize_reward=False)
    env = ENV.make(env, action_repeat=args.action_repeat, mode='mm')
    # stack several consecutive frames together

    # Hydra run_dir is set before main() to outputs/date/task_name/time.
    output_dir = hydra_output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.work_dir = output_dir

    import shutil
    shutil.copy('mjpr.py', os.path.join(output_dir, 'mjpr.py'))
    shutil.copy(__file__, os.path.join(output_dir, os.path.basename(__file__)))
    shutil.copy('config.yaml', os.path.join(output_dir, 'config.yaml'))

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = [int(args.action_shape)]

    shape_obs = env.reset()
    image_obs_shape = tuple(shape_obs[0].shape)
    position_obs_shape = tuple(shape_obs[1].shape)

    if args.encoder_type == 'pixel':
        obs_shape = [
            (args.frame_stack, args.image_size, args.image_size),
            position_obs_shape,
        ]
        pre_aug_obs_shape = [
            (args.frame_stack,
             args.pre_transform_image_size,
             args.pre_transform_image_size),
            position_obs_shape,
        ]
    else:
        obs_shape = [image_obs_shape, position_obs_shape]
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
            reward_path = os.path.join(output_dir, f'reward_{env_id}_{args.agent}_{args.seed}.npy')
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
            agent.save(env_id, args.agent, str(args.seed), step, model_dir=output_dir)
            agent.save_spr(env_id, args.agent, str(args.seed), step, model_dir=output_dir)
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
    # Finalize training and save the final auxiliary model.
    # ==========================================
    print("\nTraining finished. Saving final model...")
    final_step = args.num_env_steps
    
    # 1. Optional final evaluation.
    # L.log('eval/episode', episode, final_step)
    # r = evaluate(env, agent, args.num_eval_episodes, L, final_step)
    
    # 2. Optional final logger flush.
    # L.dump(final_step)
    
    # 3. Optional reward curve save.
    # reward_all.append(r)
    # reward_path = os.path.join(output_dir, f'reward_{env_id}_{args.agent}_{args.seed}.npy')
    # np.save(reward_path, np.array(reward_all))
    
    # 4. Save the final auxiliary model.
    # agent.save(env_id, args.agent, str(args.seed), final_step, model_dir=output_dir)
    agent.save_spr(env_id, args.agent, str(args.seed), final_step, model_dir=output_dir)
    
    # 5. Close WandB cleanly.
    wandb.finish()
    
    
    
    
    print(f"Final save complete at step {final_step}.")

if __name__ == '__main__':
    sys.argv, CLI_ENV_NAME, CLI_SEED = _extract_cli_args(sys.argv)
    RUN_SEED = _resolve_run_seed(sys.argv, CLI_SEED)
    sys.argv = _append_hydra_run_dir(sys.argv, CLI_ENV_NAME, RUN_SEED)
    torch.multiprocessing.set_start_method('spawn')
    main()
