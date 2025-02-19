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
import numpy as np
import os
from datetime import datetime
import sys
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, wandb_helper
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import torch.multiprocessing as mp
from muti_GPU_TEST.rsl_rl.rsl_rl.utils.ddp_utils import setup, cleanup
import wandb

#num_mini_batches应该 设置成能被GPU数量整除的数
#batch_size = self.num_envs * self.num_transitions_per_env应该能被GPU数量整除
def train(rank, world_size,args):
    available_gpus = torch.cuda.device_count()
   
    if world_size > available_gpus: # 检查GPU数量是否足够
        print(f"Warning: The number of processes ({world_size}) is less than the number of available GPUs ({available_gpus}).")
        sys.exit(1)
    # 初始化分布式训练环境
    if rank == 0:
     setup(rank, world_size)

    torch.cuda.set_device(rank)
    
    # 创建环境（仅主GPU需要真实环境）
    if rank == 0:
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
    else:
        # 非主GPU创建虚拟环境（占位用）
        env = None
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f'{args.task}'

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    # Check if we specified that we want to use wandb
    do_wandb = train_cfg.do_wandb if hasattr(train_cfg, 'do_wandb') else False
    # Do the logging only if wandb requirements have been fully specified
    do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)

    if do_wandb:
        wandb.config = {}

        if hasattr(train_cfg, 'wandb'):
            what_to_log = train_cfg.wandb.what_to_log
            wandb_helper.craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)

        print(f'Received WandB project name: {args.wandb_project}\nReceived WandB entitiy name: {args.wandb_entity}\n')
         # 仅主GPU初始化WandB
        if rank == 0 and args.wandb_name:
            wandb.init(project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    config=wandb.config,
                    name=experiment_name)

        ppo_runner.configure_wandb(wandb)
        ppo_runner.configure_learn(train_cfg.runner.max_iterations, True)
        ppo_runner.learn()

        wandb.finish()
    else:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations) #init_at_random_ep_len=True)
    # 每个训练进程完成后进行清理
    cleanup()
     #多GPU兼容性：在多GPU训练中，所有进程的初始状态需严格一致。如果 init_at_random_ep_len=True 会随机初始化主进程的环境步长，但其他进程无法同步此随机状态，可能导致训练不一致。
if __name__ == '__main__':
    world_size = 1
    args = get_args()
    mp.spawn(
        train,
        args= (world_size, args),
        nprocs=world_size,
        join=True
    )

#在终端设置环境变量 export MASTER_ADDR="127.0.0.1"  # 机器的 IP 地址   export MASTER_PORT="12355"  # 设置一个空闲的端口

   # train(args) # 单GPU训练

# import numpy as np
# import os
# from datetime import datetime

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# import torch

# def train(args):
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#     ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

# if __name__ == '__main__':
#     args = get_args()
#     train(args)
