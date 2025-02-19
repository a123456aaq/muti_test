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

from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union

# minimal interface of the environment
class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor 
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass
#     #这个文件定义了一个抽象类 VecEnv，描述了多环境在强化学习情境中的基本接口及其属性。它继承自 Python 的 ABC，使用 @abstractmethod 装饰器强制子类必须实现一些关键方法。VecEnv 中的属性（如 num_envs、num_obs、obs_buf、rew_buf 等）用于存储环境数量、观测数据、奖励等信息，便于多环境并行执行。接口方法包括：

# • step(actions)：执行一步环境计算，通常返回新的观测、潜在的特权观测、奖励、重置标识以及额外信息。
# • reset(env_ids)：重置指定环境（env_ids），常用于开始新回合或处理中途需要重置的环境。
# • get_observations()：获取当前环境对普通智能体可见的观测数据。
# • get_privileged_observations()：获取只有特权智能体或特定情况才可使用的观测数据，可能返回 None。

# 这些抽象方法保证具体环境实现时的统一接口，方便在多环境强化学习中进行并行处理和数据收集。