"""
    Use rpc server to receive the timestamps of commu (one worker one step one hook request)
"""
import os
import sys
import argparse
import time
import csv
import grpc
import threading
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.join(os.path.dirname(__file__), '../acclrpc'))
from protobuf import coordinator_pb2
from protobuf import coordinator_pb2_grpc

from concurrent import futures

# Environment variables set by mpirun
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

# centralized
class Coordinator(coordinator_pb2_grpc.CoordinatorServicer):
    def __init__(self, ip, port, max_step, output_file, heter_alpha) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.max_step = max_step
        self.output_file = output_file
        self.heter_alpha = heter_alpha
        # {step: [rank, ...]}
        self.record = {}
        self.record_maxmin = {}
        for i in range(self.max_step):
            self.record[i] = [0 for j in range(WORLD_SIZE)]

    def hook_fetch(self, request, context):
        step = request.step
        world_rank = request.world_rank
        self.record[step][world_rank] = time.time()
        return coordinator_pb2.hook_response(active_list=[])

    def get_record_maxmin(self, ) -> dict:
        for i in range(self.max_step):
            self.record_maxmin[i] = [
                max(self.record[i]),
                min(self.record[i])
            ]

        with open(self.output_file, 'w') as file:
            writer = csv.writer(file)
            for i in range(self.max_step):
                writer.writerow([(self.record_maxmin[i][0] - self.record_maxmin[i][1]) * self.heter_alpha])

        return self.record_maxmin


class WorkerHook:
    def __init__(self, coordinator_ip, port) -> None:
        self.coordinator_ip = coordinator_ip
        self.port = port
        self.current_step = -1
        self.local_hook_num = 0 # times of trigging hooks within one step

        channel = grpc.insecure_channel('%s:%s' % (self.coordinator_ip, self.port))
        self.stub = coordinator_pb2_grpc.CoordinatorStub(channel)
    
    def forward_step(self):
        self.current_step += 1
        self.local_hook_num = 0

    def send_ready_request(self):
        response = self.stub.hook_fetch(coordinator_pb2.hook_request(step=self.current_step, world_rank=WORLD_RANK))
        return response.active_list

    def ddp_hook(self, state: object, bucket: dist.GradBucket):
        if self.local_hook_num == 0:
            self.send_ready_request()
            self.local_hook_num += 1

        # dist.all_reduce(bucket.buffer())
        fake_tensor = torch.ones(10, dtype=torch.float32).to(LOCAL_RANK)
        dist.all_reduce(bucket.buffer())

        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())
        return fut

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--coordinator", type=str, default="127.0.0.1")
    parser.add_argument("--output", type=str, default="wait_time.csv")
    parser.add_argument("--heter_alpha", type=float, default=1.0)
    args = parser.parse_args()

    if WORLD_RANK == 0:
        coordinator = Coordinator(args.coordinator, "50051", args.steps, args.output, args.heter_alpha)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        coordinator_pb2_grpc.add_CoordinatorServicer_to_server(coordinator, server)
        server.add_insecure_port('%s:%s' % (coordinator.ip, coordinator.port))
        server.start()
    
    worker_hook = WorkerHook(args.coordinator, "50051")
    
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)

    model = models.vgg16().to(LOCAL_RANK)
    ddp_model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, bucket_cap_mb=25) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    ddp_model.register_comm_hook(state=None, hook=worker_hook.ddp_hook)

    for i in range(args.steps):
        worker_hook.forward_step()

        outputs = ddp_model(torch.randn(args.batch, 3, 224, 224).to(LOCAL_RANK)) #  8,10
        labels = torch.randint(0, 1, [args.batch]).to(LOCAL_RANK) # torch.randn(8, 10)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if WORLD_RANK == 0:
            print("======== step %d \t loss %0.3f" % (i, loss))

    if WORLD_RANK == 0:
        coordinator.get_record_maxmin()
        server.wait_for_termination()
