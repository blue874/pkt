# init_print.py
import os
import torch
import builtins
import warnings

original_print = builtins.print

def suppress_warnings_except_rank0():
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank（进程绑定的 GPU）
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数

    if rank != 0:
        # 禁用所有警告输出
        warnings.filterwarnings("ignore")


def suppress_print_except_rank0():
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank（进程绑定的 GPU）
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数


    def print_only_rank0(*args, **kwargs):
        if rank == 0:
            # 调用原始的 print 函数，避免递归
            original_print(*args, **kwargs)

    # 重写内置 print 函数
    builtins.print = print_only_rank0

suppress_print_except_rank0()
suppress_warnings_except_rank0()