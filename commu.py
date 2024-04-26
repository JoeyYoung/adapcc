"""
    Packing the dynamic library of the C bachend
    Implement the control plane
    Feed communicator plane with relay info and strategy
"""
from ctypes import *
from threading import Thread
import time
import torch.distributed as dist
import torch
import os
import sys
import grpc
import argparse
import xmltodict
from concurrent import futures
from dispatcher import Dispatcher

sys.path.append(os.path.join(os.path.dirname(__file__), './proto'))
from proto.rpc_client import Controller, Hooker
from proto.rpc_server import Coordinator
from proto.protobuf import coordinator_pb2, coordinator_pb2_grpc
from queue import Queue
sys.path.append(os.path.join(os.path.dirname(__file__), './gurobi'))
from synthesizer import *

# supported primitives
ALLREDUCE = 0
REDUCE = 1
BOARDCAST = 2
ALLGATHER = 3
ALLTOALL = 4
REDUCESCATTER = 5 
DETECT = 6
PROFILE = 7

rpc_latency_file = open("./proto/latency.txt", "w") # negotiation overhead measurement

class CudaCommu:
    def __init__(self, args, dylib, local_rank, world_rank, world_size):
        # launch cuda thread
        self.background_thread = None
        self.time_init_wait = 3
        self.args = args

        # rank info
        self.dylib = dylib
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.world_size = world_size

        # use different ports for ipc channels
        self.port = int(self.args.port)
        self.port_accum = 2000
        self.init_count = 0
        
        # read from ip_table.txt
        self.ip_table_file = "./topology/ip_table.txt"
        self.ip_table = []
        self._read_ip_table()

        # used for explict message passing
        self.dispatcher = Dispatcher(self.ip_table)

        self.prim_repo = {
            ALLREDUCE: self.dylib.allreduce,
            REDUCE: self.dylib.reduce,
            BOARDCAST: self.dylib.boardcast,
        }

        self.active_gpus = [i for i in range(self.world_size)]
        self.coordinator = None
        self.server = None # grpc server
        self.synthesizer = Synthesizer(
            strategy_file=self.args.strategy_file,
            ip_table=self.ip_table
        )
        self.chunk_bytes = None

        # we only launch the coordinator on rank 0 worker
        if world_rank == 0:
            self.coordinator = Coordinator(self.ip_table[world_rank], 50051, world_size)
            self.coordinator_thread = Thread(target=self._coordinator_thread_func, args=())
            self.coordinator_thread.start()
        
        # rpc client: send relay fetch request
        self.controller = Controller(self.ip_table[0], 50051) 
        # rpc client: send commu ready request
        self.hooker = Hooker(self.ip_table[0], 50051) 
        self.step_queue = Queue()

        # controller agent on each worker
        self.controller_thread = Thread(target=self._controller_thread_func, args=())
        self.controller_thread.start()
        self.relay_results = []
        self.relay_signal_queue = Queue()

        self.current_step = 0
        self.local_hook_num = 0

        # used for controller backend
        self.bucket_info = []
        self.relay_buffer = []
        self.bsp_queue = Queue()
        self.accumulated_bw = 0
        self.is_bsp = True
        self.fault_worker_list = []
        
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

    # push a communication request into the work queue as a work element
    def _enqueue_work_elem(self, buffer, size, chunk_bytes, active_gpus, prim):
        tensor_ptr = c_void_p(buffer.data_ptr()) 
        size = c_int(size)
        chunk_bytes = c_int(chunk_bytes)

        num_gpus = len(active_gpus)
        arr = (c_int * num_gpus)(*active_gpus)
        arr_len = c_int(num_gpus)
        self.prim_repo.get(prim)(tensor_ptr, size, chunk_bytes, arr, arr_len)
        
        return buffer

    def _coordinator_thread_func(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        coordinator_pb2_grpc.add_CoordinatorServicer_to_server(self.coordinator, self.server)
        self.server.add_insecure_port('%s:%s' % (self.coordinator.ip, self.coordinator.port))
        self.server.start()
        self.server.wait_for_termination()

    def _controller_thread_func(self):
        """
            Upon into a new step, send request to fetch relay
            For relays, trigger async op for each bucket
        """
        while True:
            step = self.step_queue.get()
            if step == -1:  break
            self.active_gpus, status = self.controller.send_relay_request(step, self.world_rank)
            # fault occurs, get fault worker list
            if status == 0:
                for w in range(self.world_size):
                    if w not in self.active_gpus: self.fault_worker_list.append(w)
                print("Fault occurs: rank %d alive" % self.world_rank)
                return
            print("[Rank %d]Controller active:" % (self.world_rank), self.active_gpus)
            if step <= 1:   continue
            if self.world_rank not in self.active_gpus:
                for i in range(len(self.bucket_info)):
                    size, chunk_bytes = self.bucket_info[i]
                    self.relay_results.append(
                        self.all_reduce(
                            self.relay_buffer[i], size, chunk_bytes, self.active_gpus
                        )
                    )
                    self.relay_signal_queue.put(step)
                
                self.bsp_queue.put(step)

    def _recursive_server_dict(self, data, layer_idx, layers):
        if layer_idx == len(layers):
            g = n = 0
            if 'gpu' in data: g += len(data['gpu'])
            if 'nic' in data:
                if isinstance(data['nic'], list): n += len(data['nic'])
                else: n += 1
            return g, n
        
        g_num = n_num = 0
        layer = layers[layer_idx]
        content = data[layer]
        if isinstance(content, dict):
            g, n = self._recursive_server_dict(content, layer_idx + 1, layers)
            g_num += g
            n_num += n
        else:
            for elem in content:
                g, n = self._recursive_server_dict(elem, layer_idx + 1, layers)
                g_num += g
                n_num += n
        return g_num, n_num

    def _get_local_rank0_list(self, ):
        # return a list including world rank ids of each local rank0
        local_rank0_list = []
        world_rank = 0
        visit = {}
        for ip in self.ip_table:
            if ip not in visit:
                local_rank0_list.append(world_rank)
                visit[ip] = True
            world_rank += 1
        return local_rank0_list

    def _gather_detect_graph(self):     
        files = []
        ips = []
        local_rank0_list = self._get_local_rank0_list()
        for rank in local_rank0_list:
            files.append('./topology/topo_detect_%d.xml' % rank)
            ips.append(self.ip_table[rank])

        servers = []
        for file in files:
            with open(file, 'r') as f:
                servers.append(xmltodict.parse(f.read()))
        
        logical_graph = {'graph': {'server': []}}
        for sid in range(len(servers)):
            server = servers[sid]
            layers = ['cpu', 'pcie']
            layer_idx = 0
            gpu_num, nic_num = self._recursive_server_dict(server, layer_idx, layers)
            gpus_per_nic = int(gpu_num / nic_num)
            server_config = {"@id": str(sid), "@ip": ips[sid]}
            if nic_num > 1:
                server_config['nic'] = []
                for n in range(nic_num):
                    nic_config = {"@id": str(n), "gpu": []}
                    for g in range(gpus_per_nic*n, gpus_per_nic*(n+1)):
                        nic_config['gpu'].append({'@id': str(g)})
                server_config['nic'].append(nic_config)
            else:
                server_config['nic'] = {"@id": "0"}
                server_config['nic']['gpu'] = []
                for g in range(gpu_num):
                    server_config['nic']['gpu'].append({"@id": str(g)})
        
            logical_graph['graph']['server'].append(server_config)

        with open(self.args.logical_graph, 'w') as f:
            f.write(xmltodict.unparse(logical_graph))

    def _gather_topo_profile(self):
        files = []
        local_rank0_list = self._get_local_rank0_list()
        for rank in local_rank0_list:
            files.append('./topology/topo_profile_%d' % rank)
        
        bw_graph = [[0 for i in range(self.world_size)] for j in range(self.world_size)]
        lc_graph = [[0 for i in range(self.world_size)] for j in range(self.world_size)]
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    elems = l[:len(l)-1].split(',')
                    src_rank = int(elems[0])
                    dst_rank = int(elems[1])
                    probe_type = int(elems[2])
                    probe_value = float(elems[3])
                    if probe_type == 0: lc_graph[src_rank][dst_rank] = probe_value
                    else:   bw_graph[src_rank][dst_rank] = probe_value

        self.accumulated_bw = 0
        for i in range(self.world_size):
            for j in range(self.world_size):
                self.accumulated_bw += (bw_graph[i][j]) / 2
        return lc_graph, bw_graph

    def _synthesis_strategy(self, lc_graph, bw_graph):
        # read profiler dump files, generate strategy
        self.synthesizer.set_ip_info(self.ip_table)
        self.synthesizer.set_parallel_degree(self.args.parallel_degree)
        self.synthesizer.set_latency_graph(lc_graph)
        self.synthesizer.set_bandwidth_graph(bw_graph)
        self.chunk_bytes = self.synthesizer.generate_strategy('reduce')
        
    """
    ======================================================================
    Exposed API for training
    ======================================================================
    """
    def clear(self):
        """
            stop the controller and grpc server
        """
        self.update_relay(-1)
        if self.world_rank == 0:
            self.server.stop(1)

    def update_relay(self, step):
        """
            trigger controller to fetch relay from coordinator
        """
        self.step_queue.put(step)
        self.current_step = step
        self.local_hook_num = 0

    def init_threads(self, prim):
        """
            Use accumulated ports for different context
            allow running differnet primitives sequentially
        """
        bootstrap_file = None
        if prim == DETECT: bootstrap_file = self.ip_table_file
        elif prim == PROFILE: bootstrap_file = self.args.logical_graph
        else: bootstrap_file = self.args.strategy_file

        ipc_channel_port = self.port + self.port_accum * self.init_count
        self.background_thread = Thread(
            target=self._init_thread_func, args=(prim, bootstrap_file, ipc_channel_port)
        )
        self.background_thread.start()
    
        # ensure the context has been setup
        time.sleep(self.time_init_wait)
        self.init_count += 1

    def exit_threads(self, prim):
        """
            automatic workflow framework
            for certain prims, some stuffs need to be done after exit
        """
        self._exit_thread_func(prim)
        self.background_thread.join()
        if prim == DETECT:
            if self.local_rank == 0:
                self.dispatcher.dispatch_detected_topo(
                    "./topology/topo_detect*", 
                    "%s/topology" % os.path.dirname(os.path.abspath(__file__))
                )
            dist.barrier()
            if self.local_rank == 0: self._gather_detect_graph()
            dist.barrier()
            
        if prim == PROFILE:
            if self.local_rank == 0:
                self.dispatcher.send_profiled_topo(
                    "./topology/topo_profile*",
                    "%s/topology" % os.path.dirname(os.path.abspath(__file__))
                )
            dist.barrier()
            if self.world_rank == 0:
                lc_graph, bw_graph = self._gather_topo_profile()
                self._synthesis_strategy(lc_graph, bw_graph)
                self.dispatcher.dispatch_strategy(
                    self.args.strategy_file,
                    "%s/strategy" % os.path.dirname(os.path.abspath(__file__))
                )
            dist.barrier()

    '''
        @buffer: torch tensor on device or host
        @size: number of floats in the tensor
        @chunk_bytes: chunk size in terms of bytes
        @active_gpus: gpu id equals to rank process id, instead of local rank;
    '''
    def all_reduce(self, buffer, size, chunk_bytes=None, active_gpus=None):
        if chunk_bytes == None: chunk_bytes = self.chunk_bytes
        if active_gpus == None: active_gpus = self.active_gpus
        return self._enqueue_work_elem(
            buffer, size, chunk_bytes, active_gpus, ALLREDUCE
        )

    def reduce(self, buffer, size, chunk_bytes=None, active_gpus=None):
        if chunk_bytes == None: chunk_bytes = self.chunk_bytes
        if active_gpus == None: active_gpus = self.active_gpus
        return self._enqueue_work_elem(
            buffer, size, chunk_bytes, active_gpus, REDUCE
        )

    def boardcast(self, buffer, size, chunk_bytes=None, active_gpus=None):
        if chunk_bytes == None: chunk_bytes = self.chunk_bytes
        if active_gpus == None: active_gpus = self.active_gpus
        return self._enqueue_work_elem(
            buffer, size, chunk_bytes, active_gpus, BOARDCAST
        )

    '''
        DDP hook registered, once the bucket ready call this func
        call allreduce, wait for Future result
    '''
    def cuda_allreduce_hook(self, state: object, bucket: dist.GradBucket):
        if self.local_hook_num == 0:
            # rpc_start = time.time()
            self.active_gpus = self.hooker.send_ready_request(
                self.current_step, self.world_rank
            )
            # rpc_end = time.time()
            # print("rpc latency:", rpc_end - rpc_start)
            # rpc_latency_file.write(str(format(rpc_end - rpc_start, 'f')))
            # rpc_latency_file.write("\n")

        self.local_hook_num += 1
        print("[Rank %d]hook active:" % (self.world_rank), self.active_gpus)

        buffer = bucket.buffer()
        size = int(buffer.size()[0])
        total_bytes = 4 * size
        if total_bytes > 10*1024*1024: chunk_bytes = 4*1024*1024
        else: chunk_bytes = int(total_bytes / 4)

        print("[Rank %d]tensor float number %d, %d bytes, chunk bytes %d" % (
            self.world_rank, size, total_bytes, chunk_bytes)
        )
        
        if self.current_step == 0:
            # only one bucket
            reduced_data = self.all_reduce(buffer, size, chunk_bytes, self.active_gpus)
        elif self.current_step == 1:
            # record bucket info
            self.bucket_info.append((size, chunk_bytes))            
            self.relay_buffer.append(
                torch.ones(size, dtype=torch.float32).to(self.local_rank)
            )
            reduced_data = self.all_reduce(buffer, size, chunk_bytes, self.active_gpus)
        else:
            # start relay
            if self.world_rank in self.active_gpus:
                reduced_data = self.all_reduce(buffer, size, chunk_bytes, self.active_gpus)
            else:
                if self.is_bsp:
                    reduced_data = bucket.buffer()
                    if self.local_hook_num == len(self.bucket_info):  self.bsp_queue.get()
                else:
                    self.dylib.updateActive(c_int(self.world_rank))
                    for i in range(size): self.relay_buffer[i] = buffer[i]
                    assert(self.relay_signal_queue.get() == self.current_step)
                    reduced_data = self.relay_results.pop(0)

        fut = torch.futures.Future()
        fut.set_result(reduced_data)
        return fut


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str)
    parser.add_argument("--strategy_file", type=str)
    parser.add_argument("--logical_graph", type=str)
    parser.add_argument("--entry_point", type=int)
    args = parser.parse_args()

    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    commu_lib = CudaCommu(
        args, CDLL('./communicator.so'), LOCAL_RANK, WORLD_RANK, WORLD_SIZE
    )

    commu_lib.init_threads(ALLREDUCE)

    for i in range(1, 3):
        tensor = torch.ones(16, dtype=torch.float32) * i
        tensor = tensor.to(LOCAL_RANK)
        size = int(tensor.size()[0])
        chunk_bytes = 8
        active_gpus = [j for j in range(WORLD_SIZE)]
        num_gpus = len(active_gpus)

        reduced_tensor = commu_lib.all_reduce(tensor, size, chunk_bytes, active_gpus)
        print("rank %d:" % (WORLD_RANK), reduced_tensor)

    rpc_latency_file.close()
    commu_lib.exit_threads(ALLREDUCE)
    commu_lib.clear()
