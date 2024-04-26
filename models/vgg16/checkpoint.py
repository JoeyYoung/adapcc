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

# Environment variables set by mpirun
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

def noncommu_hook(state:object, bucket: dist.GradBucket):
    buffer = bucket.buffer()
    fut = torch.futures.Future()
    fut.set_result(buffer)
    return fut
    

def init_processes(backend):
    # init
    start = time.time()
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)
    model = models.vgg16().to(LOCAL_RANK)
    ddp_model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    end = time.time()
    print("rank %d: model / nccl init time %.3f" % (WORLD_RANK, end - start))
    
    # checkpoint
    start = time.time()
    CHECKPOINT_PATH = "./model.checkpoint"
    if WORLD_RANK == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    dist.barrier()
    end = time.time()
    print("rank %d: checkpoint time %.3f" % (WORLD_RANK, end - start))

    # restore
    start = time.time()
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    end = time.time()
    print("rank %d: restore time %.3f" % (WORLD_RANK, end - start))
    
    batch_size = 256
    ddp_model.register_comm_hook(state=None, hook=noncommu_hook)
    
    start = time.time()
    outputs = ddp_model(torch.randn(batch_size, 3, 224, 224).to(LOCAL_RANK)) #  8,10
    labels = torch.randint(0, 1, [batch_size]).to(LOCAL_RANK) # torch.randn(8, 10)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end = time.time()
    print("rank %d: computation time %.3f" % (WORLD_RANK, end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
