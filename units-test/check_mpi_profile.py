"""
    rpc latency: 1ms, use queue for polling
    ddp first step has a very large bucket size - eager mode: go through the model first
"""

from ctypes import *
from threading import Thread
import time
import os
import sys
import argparse
from queue import Queue

# supported primitives
ALLREDUCE = 0
REDUCE = 1
BOARDCAST = 2
ALLGATHER = 3
ALLTOALL = 4
REDUCESCATTER = 5 
DETECT = 6
PROFILE = 7

class CudaCommu:
    def __init__(self, dylib, port, local_rank, world_rank, world_size):
        # launch cuda thread
        self.background_thread = None
        self.time_init_wait = 3

        # rank info
        self.dylib = dylib
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.world_size = world_size

        # use different ports for ipc channels
        self.port = port
        self.port_accum = 2000
        self.init_count = 0
        
        # read from ip_table.txt
        self.ip_table_file = "./topology/ip_table.txt"
        self.ip_table = []
        self._read_ip_table()

        self.active_gpus = [i for i in range(self.world_size)]
        
    def _read_ip_table(self):
        f = open(self.ip_table_file)
        lines = f.readlines()
        for i in range(len(lines)):
            self.ip_table.append(lines[i].replace("\n", ""))
        f.close()

    def _init_thread_func(self, prim, strategy, port):
        b_strategy = strategy.encode('utf-8')
        self.dylib.initThreads(c_int(prim), (c_char_p)(b_strategy), (c_int)(port))

    def _exit_thread_func(self, prim):
        self.dylib.exitThreads(c_int(prim))

    def init_threads(self, prim, strategy):
        ipc_channel_port = self.port + self.port_accum * self.init_count
        self.background_thread = Thread(target=self._init_thread_func, args=(prim, strategy, ipc_channel_port))
        self.background_thread.start()
    
        # ensure the context has been setup
        time.sleep(self.time_init_wait)

    def exit_threads(self, prim):
        self._exit_thread_func(prim)
        self.background_thread.join()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--logical_graph", type=str)
    args = parser.parse_args()

    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    commu_lib = CudaCommu(CDLL('./communicator.so'), int(args.port), LOCAL_RANK, WORLD_RANK, WORLD_SIZE)

    commu_lib.init_threads(PROFILE, args.logical_graph)
    commu_lib.exit_threads(PROFILE)
    time.sleep(20)
