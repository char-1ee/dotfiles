  # test_move.py
import argparse
import os
import time
from typing import Type

import torch
import torch.distributed as dist
from prettytable import PrettyTable


def init_dist():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")

    torch.cuda.set_device(local_rank)


def get_time():
    torch.cuda.synchronize()
    return time.time()


def log(*args):
    if dist.get_rank() == 0:
        print(*args)

### Communicatoin ops ###


class MoveOp:
    def __init__(self, size: int) -> None:
        self.cpu_tensor = torch.rand(size)
        self.cuda_tensor = torch.rand(size, device='cuda')

    def __call__(self) -> None:
        raise NotImplementedError


class H2D(MoveOp):
    def __call__(self) -> None:
        self.cuda_tensor.copy_(self.cpu_tensor)


class D2H(MoveOp):
    def __call__(self) -> None:
        self.cpu_tensor.copy_(self.cuda_tensor)


class PinnedH2D(H2D):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.cpu_tensor = self.cpu_tensor.pin_memory()


class PinnedD2H(D2H):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.cpu_tensor = self.cpu_tensor.pin_memory()


def collect_time(op: MoveOp, n_iters: int) -> float:
    start = get_time()
    for _ in range(n_iters):
        op()
    end = get_time()
    return (end - start) / n_iters


def benchmark(op_cls: Type[MoveOp], sizes: list, n_iters: int, n_warmup: int = 5, dtype=torch.float) -> None:
    element_size = torch.finfo(dtype).bits // 8
    sizes = sorted(sizes)
    counts = [size // element_size for size in sizes]
    # warmup for min
    op = op_cls(counts[0])
    collect_time(op, n_warmup)
    # warmup for max
    op = op_cls(counts[-1])
    collect_time(op, n_warmup)
    # benchmark
    busbw_sum = 0
    table = PrettyTable(['size(B)', 'count(elements)', 'type', 'time(ms)',
                         'busbw(GB/s)'], float_format='.2')
    for size, count in zip(sizes, counts):
        assert size % element_size == 0, "size must be divisible by element_size"
        duration = collect_time(op, n_iters)
        duration = torch.tensor([duration], device='cuda')
        dist.all_reduce(duration)
        duration.div_(dist.get_world_size())
        busbw = size / duration.item()
        busbw_sum += busbw
        table.add_row([size, count, dtype, duration.item() * 1000, busbw / 1024**3])
    avg_busbw = busbw_sum / len(sizes)
    if dist.get_rank() == 0:
        print(table)
        print(f'Average busbw: {avg_busbw/1024**3:.3f} GB/s')


def parse_size(s: str) -> int:
    s = s.upper()
    if s[-1] == 'B':
        s = s[:-1]
    if s[-1] == 'K':
        return int(s[:-1]) * 1024
    if s[-1] == 'M':
        return int(s[:-1]) * 1024**2
    if s[-1] == 'G':
        return int(s[:-1]) * 1024**3
    return int(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--begin', type=str, default='32M')
    parser.add_argument('-e', '--end', type=str, default='32M')
    parser.add_argument('-s', '--step', type=str, default='2M')
    parser.add_argument('-f', '--factor', type=int, default=1)
    parser.add_argument('-i', '--iters', type=int, default=20)
    parser.add_argument('-w', '--warmup', type=int, default=5)
    parser.add_argument('-d', '--dtype', type=str, default='float', choices=['float', 'fp16', 'bf16'])
    args = parser.parse_args()
    init_dist()
    if args.factor > 1:
        sizes = []
        start = parse_size(args.begin)
        end = parse_size(args.end)
        while start <= end:
            sizes.append(start)
            start *= args.factor
    else:
        sizes = list(range(parse_size(args.begin), parse_size(args.end) + 1, parse_size(args.step)))
    if args.dtype == 'float':
        dtype = torch.float
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    for comm_op in [H2D, D2H, PinnedH2D, PinnedD2H]:
        if dist.get_rank() == 0:
            print(f'Running {comm_op.__name__} benchmark')
        benchmark(comm_op, sizes, args.iters, args.warmup, dtype=dtype)