"""
    @ Template Training Scripts:
    Communication managed by MPI, based on known bucket info
    
    1: Link to the C library with path and port
    2: Detect with ip table & dump topo in local rank 0
    3: do profiling, systhesize strategy
    4: register the commu hook
    5: send terminal signal 
"""
import os
import argparse
from ctypes import *
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

from adapcc import *

# Environment variables set by mpirun
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

def init_processes(args):
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)

    model = models.vgg16().to(LOCAL_RANK)
    ddp_model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, bucket_cap_mb=100) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    AdapCC.init(args, LOCAL_RANK, WORLD_RANK, WORLD_SIZE)
    AdapCC.setup(ALLREDUCE)
    ddp_model.register_comm_hook(state=None, hook=AdapCC.communicator.cuda_allreduce_hook)

    for i in range(5):
        AdapCC.communicator.update_relay(step=i)
        if i != 0 and i % AdapCC.profile_freq == 0: 
            AdapCC.reconstruct_topology(args, ALLREDUCE)
        
        outputs = ddp_model(torch.randn(64, 3, 224, 224).to(LOCAL_RANK))
        labels = torch.randint(0, 1, [64]).to(LOCAL_RANK)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if WORLD_RANK == 0:
            print("======== step %d \t loss %0.3f" % (i, loss))

    AdapCC.clear(ALLREDUCE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    parser.add_argument("--port", type=str)
    parser.add_argument("--strategy_file", type=str)
    parser.add_argument("--logical_graph", type=str)
    parser.add_argument("--entry_point", type=int)
    parser.add_argument("--parallel_degree", type=int)
    parser.add_argument("--profile_freq", type=int)
    args = parser.parse_args()

    init_processes(args)
