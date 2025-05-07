import cupy as cp
import numpy as np
from .mma import mmasub as mmasub_cpu
from .mma_gpu import mmasub as mmasub_gpu

def mmasub(*args, np=np,**kwargs):
    if np == cp:
        return mmasub_gpu(*args, **kwargs)
    else:
        return mmasub_cpu(*args, **kwargs)