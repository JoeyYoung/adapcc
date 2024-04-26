"""
    pip install vit-pytorch
    pip install torchsummary
"""

import os
import argparse
from ctypes import *
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from torchsummary import summary
from vit_pytorch import ViT

LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)

    print("%d:%d out of %d ranks" % (LOCAL_RANK, WORLD_RANK, WORLD_SIZE))

    model = ViT(
                image_size = 256,
                patch_size = 32,
                num_classes = 1000,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            ).to(LOCAL_RANK)
    
    # summary(model, input_size=(3, 256, 256), batch_size=-1)

    ddp_model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    batch_size = 256

    while True:
        start = time.time()

        outputs = ddp_model(torch.randn(batch_size, 3, 224, 224).to(LOCAL_RANK)) #  8,10
        labels = torch.randint(0, 1, [batch_size]).to(LOCAL_RANK) # torch.randn(8, 10)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if WORLD_RANK == 0:
            print("======== step %d \t loss %0.3f" % (i, loss))

        end = time.time()
        print("computation time: %.3f" % (end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
