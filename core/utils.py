import os, random, numpy as np
try:
    import torch
except Exception:
    torch = None

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def world_info_from_env():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local_rank
