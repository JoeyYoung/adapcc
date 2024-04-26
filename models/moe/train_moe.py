"""
    From fast moe
    https://github.com/fastmoe/fastmoe

    MOE MLP, split experts over GPUs
    all-to-all for tokens dispatch and results combine
"""

import argparse
import builtins
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from torchsummary import summary

import fmoe
from fmoe import FMoETransformerMLP

import time

LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

if __name__ == '__main__':
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)

    print("%d:%d out of %d ranks" % (LOCAL_RANK, WORLD_RANK, WORLD_SIZE))

    # wrap MLP as expert, which has two linear module: d_model - input?, d_hidden - hidden
    # num_expert: # of experts each worker, no effect on computation workload
    model = FMoETransformerMLP(num_expert=10, d_model=1024, d_hidden=4096, top_k=1).to(LOCAL_RANK)
    summary(model, input_size=(3, 256, 256), batch_size=-1)

    # set model_moe
    model = fmoe.DistributedGroupedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    model._sync_params()

    batch_size = 32

    # do inference
    for i in range(20):
        start = time.time()

        x = torch.rand([batch_size, 1024, 10, 10]).to(LOCAL_RANK)
        y = model(x)

        end = time.time()
        print("computation time: %.3f" % (end - start))