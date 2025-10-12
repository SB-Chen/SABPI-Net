import os

import torch
import torch.distributed as dist
import datetime

def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    print(f"rank:{ args.rank}, world_size:{args.world_size}, gpu:{args.gpu}")
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank,
                            timeout=datetime.timedelta(seconds=1200))
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2: 
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def gather_list(value):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    value = value.cuda()
    if value.is_sparse:
        value = value.to_dense()
    world_size = dist.get_world_size()
    if world_size < 2:
        return value

    value_gather_list = [torch.zeros_like(value) for _ in range(world_size)]
    for i in range(len(value_gather_list)):
        value_gather_list[i] = value_gather_list[i].cuda()
        if value_gather_list[i].is_sparse:
            value_gather_list[i] = value_gather_list[i].to_dense()

    dist.all_gather(value_gather_list, value)
    value = torch.cat(value_gather_list, dim=0)
    return value
