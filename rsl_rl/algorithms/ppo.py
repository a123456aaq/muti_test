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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
import torch.distributed as dist
class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 rank,
                 world_size=1, 
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

         # 动态获取 rank 和 world_size
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = rank
            self.world_size = world_size   
           
        # 设备绑定
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu") 
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        # # DDP模型
        # # if not hasattr(actor_critic, 'is_moved_to_device'):
        actor_critic = actor_critic.to(self.device)
        # #     actor_critic.module.is_moved_to_device = True  # 标记模型已移动


        # # 根据 world_size 决定是否使用 DDP
        # # if world_size > 0:
        #self.actor_critic = DDP(actor_critic, device_ids=[self.rank], output_device=rank)
        
        # else:
        self.actor_critic = actor_critic
      
        self.optimizer = optim.Adam(self.actor_critic.module.parameters(), lr=learning_rate)
        # PPO components
        # self.actor_critic = actor_critic
        # self.actor_critic1.to(self.device)
        self.storage = None # initialized later
        # self.optimizer = optim.Adam(self.actor_critic.module.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, device=torch.device(f"cuda:{self.rank}") )
   
    def test_mode(self):
        self.actor_critic.module.test()
    
    def train_mode(self):
        self.actor_critic.module.train()

    def act(self, obs, critic_obs):
        # if self.actor_critic.module.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.module.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.module.act(obs).detach() #单GPU用actor_critic.module.act 多GPU用actor_critic.module.module.act
        self.transition.values = self.actor_critic.module.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.module.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.module.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.module.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.module.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.module.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
        # 同步数据到所有GPU
        self.storage.synchronize_data() 

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # 获取全局world_size（GPU总数）
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.actor_critic.module.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.module.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.module.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.module.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.module.action_mean
                sigma_batch = self.actor_critic.module.action_std
                entropy_batch = self.actor_critic.module.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                         # 同步所有GPU的kl_mean
                        kl_mean_tensor = torch.tensor(kl_mean, device=self.device)
                        dist.all_reduce(kl_mean_tensor, op=dist.ReduceOp.SUM)
                        kl_mean = kl_mean_tensor.item() / world_size  # 全局平均kl_mean
                        # 根据全局KL调整学习率
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.module.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
        # 这里将各 rank 的累加值进行 all_reduce 做总和，再平均
        mean_value_loss_t = torch.tensor(mean_value_loss, device=f'cuda:{self.rank}', dtype=torch.float32)
        mean_surrogate_loss_t = torch.tensor(mean_surrogate_loss, device=f'cuda:{self.rank}', dtype=torch.float32)
        # 使用 all_reduce 汇总所有GPU的损失值
        if dist.is_initialized():
            dist.all_reduce(mean_value_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_surrogate_loss_t, op=dist.ReduceOp.SUM)

        # 计算全局平均损失
        num_updates = self.num_learning_epochs * (self.num_mini_batches// self.world_size)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        mean_value_loss = mean_value_loss_t.item() / (num_updates * world_size)
        mean_surrogate_loss = mean_surrogate_loss_t.item() / (num_updates * world_size)

        # 清空存储
        self.storage.clear()

        # 返回全局平均损失
        return mean_value_loss, mean_surrogate_loss
