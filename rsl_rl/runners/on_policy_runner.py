# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from ..utils.ddp_utils import setup, cleanup

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 rank=0,  # # 这里是默认值，但实际值应由分布式训练环境传递
                 world_size=1,  # 新增
                 log_dir=None,
                 device='cpu'):

       # 初始化分布式环境
    
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rank = rank # rank 应该通过外部传入
        self.world_size = world_size
        #self.device = device
        self.device = torch.device(f'cuda:{rank}')  # 绑定到当前GPU
        self.env = env
        # setup(self.rank, self.world_size)  # 初始化分布式环境
        # torch.cuda.set_device(self.rank)  # 确保当前进程在对应的GPU上运行
        
        # 主GPU负责初始化环境
        if self.rank == 0:
            self.env = env
            self.env.reset()
            obs = self.env.get_observations()
            privileged_obs = self.env.get_privileged_observations()
        else:
            self.env = None
            obs = torch.zeros(self.env.num_envs, self.env.num_obs, device=self.device)
            privileged_obs = torch.zeros(self.env.num_envs, self.env.num_privileged_obs, device=self.device)

        # 广播环境参数到所有GPU
        if privileged_obs is not None:
            env_params = (self.env.num_envs, self.env.num_obs, self.env.num_privileged_obs, self.env.num_actions)
            dist.broadcast(torch.tensor(env_params, device=self.device), src=0)
        else:
            env_params = (self.env.num_envs, self.env.num_obs, self.env.num_obs, self.env.num_actions)
            dist.broadcast(torch.tensor(env_params, device=self.device), src=0)

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        # 将模型移动到设备（仅在未移动时执行）
        # if not hasattr(actor_critic, 'is_moved_to_device'):
        #actor_critic = actor_critic.to(self.device)
        #     actor_critic.module.is_moved_to_device = True  # 标记模型已移动

        if world_size == 0:
            self.actor_critic = actor_critic
        #    self.actor_critic.module = actor_critic
        else:
            self.actor_critic = DDP(actor_critic, device_ids=[rank], output_device=rank)
            
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(self.actor_critic,rank=rank, world_size=world_size, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        # 仅主GPU记录日志
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if (rank == 0 and log_dir) else None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 仅主GPU初始化环境参数
        # initialize writer
        
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if self.rank == 0 and init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        # 初始化观测
        if self.rank == 0:
            obs = self.env.get_observations()
            obs = obs.to(self.device)
            privileged_obs = self.env.get_privileged_observations()
            if privileged_obs is not None:
                privileged_obs = privileged_obs.to(self.device)
        else:
            obs = torch.zeros_like(obs)
            privileged_obs = None
        # 广播初始观测到所有GPU
        #数据广播：将 src=0（主GPU）的 obs 张量复制到所有其他GPU。
        dist.broadcast(obs, src=0)
        if privileged_obs is not None:
           dist.broadcast(privileged_obs, src=0)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.alg.actor_critic.module.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
                # ==================== 数据收集阶段 ====================
            with torch.inference_mode():
                if self.rank == 0:  # 仅主GPU与环境交互
                    for i in range(self.num_steps_per_env):
                        # 使用DDP模型的原始模块获取动作
                        actions = self.alg.act(obs, critic_obs)
                        # 原代码（单GPU，直接调用）actions = self.alg.act(obs, critic_obs)
                        # 环境交互
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                        # 转换数据到设备
                        obs = obs.to(self.device)
                        rewards = rewards.to(self.device)
                        dones = dones.to(self.device)
                        critic_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs
                        # 存储经验数据
                        self.alg.process_env_step(rewards, dones, infos)
                        if self.log_dir is not None:
                        # Book keeping
                            if 'episode' in infos:
                                ep_infos.append(infos['episode'])
                            cur_reward_sum += rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                    # 广播经验数据到所有GPU
                    self.alg.storage.synchronize_data()  # 需确保RolloutStorage实现同步方法
                else:
                    # 非主GPU等待数据同步
                    self.alg.storage.synchronize_data()
                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            if self.rank == 0:
                if self.log_dir is not None:
                    self.log(locals()) # 记录日志
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.rank == 0:
         # 在训练结束后保存模型
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
     
    def log(self, locs, width=80, pad=35):
        # 确保仅主GPU记录日志
        if self.rank == 0:
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += locs['collection_time'] + locs['learn_time']
            iteration_time = locs['collection_time'] + locs['learn_time']

            ep_string = f''
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            mean_std = self.alg.actor_critic.module.std.mean()
            fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

            self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
            self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
            if len(locs['rewbuffer']) > 0:
                self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

            str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

            if len(locs['rewbuffer']) > 0:
                log_string = (f"""{'#' * width}\n"""
                            f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                            f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                            f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
            else:
                log_string = (f"""{'#' * width}\n"""
                            f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

            log_string += ep_string
            log_string += (f"""{'-' * width}\n"""
                        f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                        f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                        f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                        f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                                locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
            print(log_string)

    def save(self, path, infos=None):
     if self.rank == 0: # 仅主GPU保存模型 
        torch.save({
            'model_state_dict': self.alg.actor_critic.module.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
            # 仅主GPU加载模型
        if self.rank == 0:
            loaded_dict = torch.load(path, map_location=self.device)
            self.alg.actor_critic.module.load_state_dict(loaded_dict['model_state_dict'])
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.current_learning_iteration = loaded_dict['iter']
        
        # 使用广播将模型参数同步到其他GPU
        dist.barrier()  # 确保主GPU加载完模型后，其他GPU再开始同步
        if self.rank != 0:
            # 其他GPU接收主GPU的模型
            self.alg.actor_critic.module.load_state_dict(loaded_dict['model_state_dict'])
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        return loaded_dict['infos']
    # def load(self, path, load_optimizer=True):
    #     loaded_dict = torch.load(path)
    #     self.alg.actor_critic.module.load_state_dict(loaded_dict['model_state_dict'])
    #     if load_optimizer:
    #         self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
    #     self.current_learning_iteration = loaded_dict['iter']
    #     return loaded_dict['infos']

    def get_inference_policy(self, device=None): 
        self.alg.actor_critic.module.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.module.act_inference
