'''
    High level exposed AdapCC APIs
'''
from commu import *

class AdapCC:
    # meta info since the first registered
    communicator_path = './communicator.so'
    communicator = None
    local_rank = None
    world_rank = None
    world_size = None
    profile_freq = None
    
    @classmethod
    def init(cls, args, local_rank, world_rank, world_size):
        from ctypes import CDLL
        cls.communicator = CudaCommu(
            args,
            CDLL(cls.communicator_path),
            local_rank,
            world_rank,
            world_size
        )
        cls.local_rank = local_rank
        cls.world_rank = world_rank
        cls.world_size = world_size
        cls.profile_freq = args.profile_freq

        if args.entry_point == DETECT:
            cls.communicator.init_threads(DETECT)
            cls.communicator.exit_threads(DETECT)
            cls.communicator.init_threads(PROFILE)
            cls.communicator.exit_threads(PROFILE)
        elif args.entry_point == PROFILE:
            cls.communicator.init_threads(PROFILE)
            cls.communicator.exit_threads(PROFILE)
        elif args.entry_point == -1:
            pass
        else:
            print("no supported entry point for init.")

    @classmethod
    def setup(cls, prim):
        cls.communicator.init_threads(prim)

    @classmethod
    def allreduce(cls, tensor, size, chunk_bytes, active_gpus):
        cls.communicator.all_reduce(tensor, size, chunk_bytes, active_gpus)    
    
    @classmethod
    def reduce(cls, tensor, size, chunk_bytes, active_gpus):
        cls.communicator.reduce(tensor, size, chunk_bytes, active_gpus)

    @classmethod
    def boardcast(cls, tensor, size, chunk_bytes):
        cls.communicator.boardcast(tensor, size, chunk_bytes)

    @classmethod
    def alltoall(cls, tensor, size, chunk_bytes):
        cls.communicator.alltoall(tensor, size, chunk_bytes)

    @classmethod
    def reconstruct_topology(cls, args, prim):
        cls.clear(prim)
        cls.init(args, cls.local_rank, cls.world_rank, cls.world_size)
        cls.setup(prim)

    @classmethod
    def set_profile_freq(cls, freq):
        cls.profile_freq = freq

    @classmethod
    def clear(cls, prim):
        cls.communicator.exit_threads(prim)
        cls.communicator.clear()

'''
    Used as benchmark templates
'''
if __name__ == "__main__":
    import torch.distributed as dist
    import torch
    import os
    import sys
    import grpc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str)
    parser.add_argument("--strategy_file", type=str)
    parser.add_argument("--logical_graph", type=str)
    parser.add_argument("--entry_point", type=int)
    parser.add_argument("--parallel_degree", type=int)
    parser.add_argument("--profile_freq", type=int)
    args = parser.parse_args()

    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    dist.init_process_group('nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    AdapCC.init(args, LOCAL_RANK, WORLD_RANK, WORLD_SIZE)
    AdapCC.setup(ALLREDUCE)

    for i in range(1, 3):
        tensor = torch.ones(16, dtype=torch.float32) * i
        tensor = tensor.to(LOCAL_RANK)
        size = int(tensor.size()[0])
        chunk_bytes = 8
        active_gpus = [j for j in range(WORLD_SIZE)]
        num_gpus = len(active_gpus)

        reduced_tensor = AdapCC.communicator.all_reduce(tensor, size, chunk_bytes, active_gpus)
        print("rank %d:" % (WORLD_RANK), reduced_tensor) # .cpu().numpy().tolist()

    AdapCC.clear(ALLREDUCE)
