# ddp_utils.py
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 主节点地址
    os.environ['MASTER_PORT'] = '12355'  # 主节点端口
    """ 初始化分布式训练环境 """
    dist.init_process_group(
        backend='nccl', # 选择后端用于通信
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    """ 清理分布式环境 """
    dist.destroy_process_group()

#参数 rank 表示当前进程的排名（ID）。
#参数 world_size 表示总进程数。