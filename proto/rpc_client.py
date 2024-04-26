from __future__ import print_function

import grpc
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), './protobuf'))
from protobuf import coordinator_pb2
from protobuf import coordinator_pb2_grpc

class Controller:
    def __init__(self, coordinator_ip, port) -> None:
        self.coordinator_ip = coordinator_ip
        self.port = port

        channel = grpc.insecure_channel('%s:%s' % (self.coordinator_ip, self.port))
        self.stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    
    def send_relay_request(self, step, world_rank):
        response = self.stub.controller_fetch(coordinator_pb2.cont_request(step=step, world_rank=world_rank))

        return response.active_list, response.status

class Hooker:
    def __init__(self, coordinator_ip, port) -> None:
        self.coordinator_ip = coordinator_ip
        self.port = port

        channel = grpc.insecure_channel('%s:%s' % (self.coordinator_ip, self.port))
        self.stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    
    def send_ready_request(self, step, world_rank):
        response = self.stub.hook_fetch(coordinator_pb2.hook_request(step=step, world_rank=world_rank))

        return response.active_list

if __name__ == '__main__':
    controller = Controller("127.0.0.1", 50051)
    hooker = Hooker("127.0.0.1", 50051)

    for i in range(20):
        controller.send_relay_request(i)
        hooker.send_ready_request(i)
