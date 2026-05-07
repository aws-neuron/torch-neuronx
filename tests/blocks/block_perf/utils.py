import torch.distributed as dist


def printf(string, rank_0_only=True):
    if dist.is_initialized() and rank_0_only and dist.get_rank() != 0:
        return
    print(string, flush=True)
