import copy
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hydra
import numpy as np
import torch
from logger import Logger
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from hydra.core.hydra_config import HydraConfig
WEBOTS_ROOT = next(parent for parent in Path(__file__).resolve().parents
                   if (parent / 'env' / 'env5.py').is_file()
                   and (parent / 'cfg' / 'config.yaml').is_file())
if str(WEBOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBOTS_ROOT))
from mjpr import mjpr
from env import env5

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

    def __init__(self, obs_shape1, obs_shape2, action_shape, capacity, batch_size, device, image_size=84, transform=None, auxiliary_task_batch_size=64, spr_horizon=5, spr_action_skip=1, time_offset=0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.spr_horizon = int(spr_horizon)
        self.spr_action_skip = int(spr_action_skip)
        self.time_offset = int(time_offset)
        if self.spr_horizon < 1:
            raise ValueError('spr_horizon must be at least 1.')
        if self.spr_action_skip < 1:
            raise ValueError('spr_action_skip must be at least 1.')
        if self.time_offset < 0:
            raise ValueError('time_offset must be non-negative.')
        self.spr_sequence_steps = self.time_offset + self.spr_horizon * self.spr_action_skip
        obs_dtype = np.float32
        self.image_obses = np.empty((capacity, *obs_shape1), dtype=obs_dtype)
        self.pos_obses = np.empty((capacity, *obs_shape2), dtype=obs_dtype)
        self.next_image_obses = np.empty((capacity, *obs_shape1), dtype=obs_dtype)
        self.next_pos_obses = np.empty((capacity, *obs_shape2), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.real_dones = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.last_save = 0
        self.full = False
        self.current_auxiliary_batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.image_obses[self.idx], obs[0])
        np.copyto(self.pos_obses[self.idx], obs[1])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_image_obses[self.idx], next_obs[0])
        np.copyto(self.next_pos_obses[self.idx], next_obs[1])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.real_dones[self.idx], done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
        image_obses = self.image_obses[idxs]
        pos_obses = self.pos_obses[idxs]
        next_image_obses = self.next_image_obses[idxs]
        next_pos_obses = self.next_pos_obses[idxs]
        image_obses = torch.as_tensor(image_obses, device=self.device).float()
        pos_obses = torch.as_tensor(pos_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_image_obses = torch.as_tensor(next_image_obses, device=self.device).float()
        next_pos_obses = torch.as_tensor(next_pos_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return ([image_obses, pos_obses], actions, rewards, [next_image_obses, next_pos_obses], not_dones)

    def sample_spr(self):
        sequence_length = self.spr_sequence_steps + 1
        available = self.capacity if self.full else self.idx
        upper = available - self.spr_sequence_steps
        if upper <= 0:
            raise ValueError(f'Not enough replay data for sample_spr: need {sequence_length} contiguous observations, currently have {available}.')
        logical_base_idxs = np.random.randint(0, upper, size=self.auxiliary_task_batch_size * 2).reshape(-1, 1)
        logical_idxs = logical_base_idxs + np.arange(sequence_length).reshape(1, -1)
        if self.full:
            idxs = (self.idx + logical_idxs) % self.capacity
        else:
            idxs = logical_idxs
        real_dones = torch.as_tensor(self.real_dones[idxs[:, :-1]], device=self.device)
        valid_idxs = torch.where((real_dones.sum(1) == 0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs]
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]
        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        pos_obses = torch.as_tensor(self.pos_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        spr_samples = {'image': image_obses.transpose(0, 1).unsqueeze(3), 'pos': pos_obses.transpose(0, 1), 'action': actions.transpose(0, 1), 'reward': rewards.transpose(0, 1)}
        return (*self.sample(), spr_samples)

    def sample(self):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        pos_obses = torch.as_tensor(self.pos_obses[idxs], device=self.device).float()
        next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        next_pos_obses = torch.as_tensor(self.next_pos_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return ([image_obses, pos_obses], actions, rewards, [next_image_obses, next_pos_obses], not_dones)

    def add_agent(self, agent):
        if hasattr(agent, 'SPR'):
            self.SPR = agent.SPR

class eval_mode(object):

    def __init__(self, *models):
        expanded = []
        for m in models:
            if m is None:
                continue
            if hasattr(m, 'actor') and hasattr(m, 'critic'):
                sub = [m.actor, m.critic, m.critic_target]
                if hasattr(m, 'SPR') and m.SPR is not None:
                    sub.append(m.SPR)
                expanded.extend(sub)
            else:
                expanded.append(m)
        self.models = expanded

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
            target = False
            episode_reward = 0
            while not (done or target):
                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, target = env.step(action)
                episode_reward += reward
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
        print('eval:', mean_ep_reward, best_ep_reward)
        return mean_ep_reward
    return run_eval_loop(sample_stochastically=False)

def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'mjpr':
        return mjpr(obs_shape=obs_shape, action_shape=action_shape, device=device, transition_model_type=args.transition_model_type, transition_model_layer_width=args.transition_model_layer_width, spr_horizon=args.spr_horizon, spr_action_skip=args.spr_action_skip, latent_dim=args.latent_dim, time_offset=args.time_offset, momentum_tau=args.momentum_tau, aug_prob=args.aug_prob, auxiliary_task_lr=args.auxiliary_task_lr, hidden_dim=args.hidden_dim, discount=args.discount, init_temperature=args.init_temperature, alpha_lr=args.alpha_lr, alpha_beta=args.alpha_beta, actor_lr=args.actor_lr, actor_beta=args.actor_beta, actor_log_std_min=args.actor_log_std_min, actor_log_std_max=args.actor_log_std_max, actor_update_freq=args.actor_update_freq, critic_lr=args.critic_lr, critic_beta=args.critic_beta, critic_tau=args.critic_tau, critic_target_update_freq=args.critic_target_update_freq, encoder_type=args.encoder_type, encoder_feature_dim=args.encoder_feature_dim, encoder_lr=args.encoder_lr, encoder_tau=args.encoder_tau, num_layers=args.num_layers, num_filters=args.num_filters, log_interval=args.log_interval, detach_encoder=args.detach_encoder, curl_latent_dim=args.curl_latent_dim)
    else:
        raise ValueError('agent is not supported: %s' % args.agent)

@hydra.main(version_base=None, config_path=str(WEBOTS_ROOT / 'cfg'), config_name='config')
def main(args: DictConfig) -> None:
    print('✅ 成功读取 config.yaml 配置文件！Hydra 解析完成。')
    reward_all = []
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    args.domain_name = args.domain_name or args.env_name.split('/')[0]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    agent_aliases = {'mib': 'mjpr'}
    if hasattr(args, 'agent') and args.agent in agent_aliases:
        print(f"Overriding agent '{args.agent}' -> '{agent_aliases[args.agent]}'")
        args.agent = agent_aliases[args.agent]
    torch.cuda.set_device(args.gpuid)
    set_seed_everywhere(args.seed)
    seed = int(args.seed)
    env = env5.env(seed, args.task_step)
    ts = time.gmtime()
    ts = time.strftime('%m%dT%H', ts)
    exp_name = 'ETA-w2'
    env_name = 'env5'
    exp_name = env_name + ts + '-im' + str(args.image_size) + '-b' + str(args.batch_size) + '-s' + str(args.seed) + '-' + args.agent
    if args.agent == 'spr_sac':
        exp_name = exp_name + '-h' + str(args.spr_horizon) + '-skip' + str(args.spr_action_skip)
    if args.agent == 'cycdm_sac':
        exp_name = exp_name + '-' + args.cycle_mode
    args.work_dir = args.work_dir + '/' + f'{args.agent}_dmc' + '/' + exp_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        total_episodes = int(math.ceil(float(args.num_env_steps) / float(args.task_step)))
    except Exception:
        total_episodes = 0
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    print(f'[DEBUG] Hydra run_dir: {run_dir}')
    source_mjpr = Path(__file__).resolve().parent / 'mjpr.py'
    output_mjpr = Path(run_dir) / 'mjpr.py'
    shutil.copy2(source_mjpr, output_mjpr)
    action_shape = [int(args.action_shape)]
    if args.encoder_type == 'pixel':
        obs_shape = (args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape
    replay_buffer = ReplayBuffer(obs_shape1=pre_aug_obs_shape, obs_shape2=args.pos_dim, action_shape=action_shape, capacity=args.replay_buffer_capacity, batch_size=args.batch_size, device=device, image_size=args.image_size, auxiliary_task_batch_size=args.auxiliary_task_batch_size, spr_horizon=args.spr_horizon, spr_action_skip=args.spr_action_skip, time_offset=args.time_offset)
    agent = make_agent(obs_shape=obs_shape, action_shape=action_shape, args=args, device=device)
    if agent is None:
        raise RuntimeError(f"make_agent returned None for agent '{args.agent}'")
    replay_buffer.add_agent(agent)
    L = Logger(run_dir, use_tb=args.save_tb, use_wandb=args.wandb)
    episode, episode_reward, done = (0, 0, True)
    training_start_time = time.time()
    start_time = time.time()
    eval_num = 0
    eval_num += 1
    r = evaluate(env, agent, args.num_eval_episodes, L, 0)
    reward_all.append(r)
    np.save(os.path.join(run_dir, 'reward_' + 'env5' + '_' + str(args.agent) + 'seed' + str(args.seed) + '.npy'), np.array(reward_all))
    for step in range(0, args.num_env_steps, args.action_repeat):
        if step > 0 and step % args.log_interval == 0:
            L.log('train/duration', time.time() - start_time, step)
            L.dump(step)
            print('duration', time.time() - start_time, step)
            start_time = time.time()
            L.log('train/episode_reward', episode_reward, step)
        if done or target:
            if step / args.eval_freq >= eval_num:
                eval_num += 1
                r = evaluate(env, agent, args.num_eval_episodes, L, step)
                reward_all.append(r)
                np.save(os.path.join(run_dir, 'reward_' + 'env5' + '_' + str(args.agent) + 'seed' + str(args.seed) + '.npy'), np.array(reward_all))
            display_episode = episode + 1
            if display_episode % args.save_model_freq == 0:
                models_dir = os.path.join(run_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                print(f'[SAVE] Episode {display_episode}: saving to {models_dir}')
                agent.save('env5', args.agent, str(args.seed), display_episode, models_dir)
                agent.save_spr('env5', args.agent, str(args.seed), display_episode, models_dir)
                import glob as glob_module
                saved_files = glob_module.glob(os.path.join(models_dir, '*.pt'))
                print(f'[SAVE] Files in {models_dir}: {len(saved_files)} files - {saved_files}')
            print(f'reward {episode_reward} step {step}')
            obs = env.reset()
            done = False
            target = False
            episode_reward = 0
            episode += 1
        if step < args.init_steps:
            action = np.random.rand(args.action_shape) * 2 - 1
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
        next_obs, reward, done, target = env.step(action)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, target)
        obs = next_obs
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    hours, remainder = divmod(int(training_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    summary_file = os.path.join(run_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f'Training Summary\n')
        f.write(f'===============\n')
        f.write(f'Total Duration: {duration_str} (hh:mm:ss)\n')
        f.write(f'Total Seconds: {training_duration:.2f}\n')
        f.write(f'Total Episodes: {episode}\n')
        f.write(f'Total Steps: {step}\n')
        f.write(f'Agent: {args.agent}\n')
        f.write(f'Seed: {args.seed}\n')
    print(f'\n[TRAINING COMPLETE] Total duration: {duration_str}\n')
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
