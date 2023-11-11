import random

import numpy as np
import torch


GENERATOR = None

def ensure(seed: str = 0xCAFE):
    global GENERATOR

    if GENERATOR is None:
        GENERATOR = torch.Generator()
        GENERATOR.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
