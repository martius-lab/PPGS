import random

import os
import numpy as np
import torch
from .settings import *


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


def efficient_from_numpy(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


def maybe_update(d, new_d):
    d.update({k: d[k] + [new_d[k]] if k in d else [new_d[k]] for k in new_d})
    return d
