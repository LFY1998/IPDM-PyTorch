import os
import torch
import numpy
os.environ['path'] += ';E:/Liaofeiyang/libtorch-win-shared-with-deps-1.7.1+cu110/libtorch/lib;D:/Anaconda3/envs/diffusion;D:/Anaconda3/envs/diffusion/Lib/site-packages/torch/lib'
def recons_torch(h_proj:torch.Tensor,lut_area:numpy.ndarray,betas:numpy.ndarray, nstart:int,ntv:int,sample_rate:int, permute:bool) ->torch.Tensor:
    """

    :param permute:
    :param sample_rate:
    :param h_proj: [BS,H,W],BS>=1
    :param lut_area:
    :param betas:
    :param nstart: number of iteration
    :param ntv:  number of total variation
    """
    ...

def proj_torch(h_volume:torch.Tensor,lut_area:numpy.ndarray,betas:numpy.ndarray)->torch.Tensor:
    """
    :param h_volume: [BS,H,W],BS>=1
    :param lut_area:
    :param betas:
    """