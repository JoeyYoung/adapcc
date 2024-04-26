"""
    coordinator only used for notifying active gpus, not directly apply
"""

from concurrent import futures

import grpc
import os
import sys
import time
from queue import Queue
import collections

sys.path.append(os.path.join(os.path.dirname(__file__), './protobuf'))
from protobuf import coordinator_pb2
from protobuf import coordinator_pb2_grpc
import time

# centralized
class Coordinator(coordinator_pb2_grpc.CoordinatorServicer):
    def __init__(self, ip, port, world_size) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.world_size = world_size

        self.relay_threshold = 0.1 # s

        # control each step
        max_steps = 1000000
        self.ready_gpus = {}
        self.response_flag = {}
        for i in range(max_steps):
            self.ready_gpus[i] = []
            self.response_flag[i] = False
        
        # use producer - consumer, instead of while polling
        self.controller_queue = Queue()
        self.hook_queue = Queue()

        self.accumulated_size = 100 * 8 / 1024 
        self.accumulated_bandwidth = 50 * self.world_size
        self.time_slot_duration = 0.005 # s
        self.worker_heartbeats_count = collections.defaultdict(list)
        self.polling_time = 0.001
        self.fault_tolerant_time = 10

    def controller_fetch(self, request, context):
        step = request.step
        self.worker_heartbeats_count[step].append(request.world_rank)
        
        # fault detection, the communication keeps processing until the next iteration
        start_time = time.time()
        while len(self.worker_heartbeats_count[step]) != self.world_size:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.fault_tolerant_time:
                # return the list of workers that are not faults
                return coordinator_pb2.cont_response(active_list=self.worker_heartbeats_count[step], status=0)
            time.sleep(self.polling_time)

        assert(self.controller_queue.get() == step)
        return coordinator_pb2.cont_response(active_list=self.ready_gpus[step], status=1)

    def hook_fetch(self, request, context):
        step = request.step
        world_rank = request.world_rank

        # fatest worker
        if len(self.ready_gpus[step]) == 0:
            self.ready_gpus[step].append(world_rank)
            inital_rent_cost = 2*(self.world_size-1)*self.accumulated_size \
                                    / self.accumulated_bandwidth
            accumulated_rent_cost = 0
            while True:
                num_readys = len(self.ready_gpus[step])
                if num_readys > 1: 
                    co_effi_n = (self.world_size - 1) / self.world_size
                    co_effi_m = (num_readys - 1) / num_readys
                    co_effi = co_effi_m / co_effi_n
                    buy_cost = inital_rent_cost * co_effi + \
                                self.world_size * self.accumulated_size / self.accumulated_bandwidth
                    
                    if (accumulated_rent_cost + inital_rent_cost) >= buy_cost or \
                        accumulated_rent_cost > self.relay_threshold or \
                            num_readys == self.world_size:
                        break
                accumulated_rent_cost += self.time_slot_duration
                time.sleep(self.time_slot_duration)

            self.response_flag[step] = True

            for i in range(len(self.ready_gpus[step]) - 1): self.hook_queue.put(step)
            for i in range(self.world_size):  self.controller_queue.put(step)
            
            active_list = [i for i in self.ready_gpus[step]] # copy
            return coordinator_pb2.hook_response(active_list=active_list)
        else:
            # relay worker
            if self.response_flag[step]:
                active_list = [i for i in self.ready_gpus[step]]
                return coordinator_pb2.hook_response(active_list=active_list)
            # active waiting worker
            else:
                self.ready_gpus[step].append(world_rank)
                assert(self.hook_queue.get() == step)

                active_list = [i for i in self.ready_gpus[step]]
                return coordinator_pb2.hook_response(active_list=active_list)
    

if __name__ == '__main__':
    coordinator = Coordinator("127.0.0.1", "50051", world_size=4)
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    coordinator_pb2_grpc.add_CoordinatorServicer_to_server(coordinator, server)
    server.add_insecure_port('%s:%s' % (coordinator.ip, coordinator.port))
    server.start()
    server.wait_for_termination()
