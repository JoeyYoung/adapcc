import time, os
from solver import Solver
from trees import ParTrees

class Synthesizer:
    def __init__(self, strategy_file, ip_table=None, parallel_degree=4, size=10*(10**6),
                    bandwidth_graph=None, latency_graph=None, policy="par-trees"):
        self.strategy_file = strategy_file
        self.ip_table = ip_table
        self.local_rank0_list = self._get_local_rank0_list()
        self.parallel_degree = parallel_degree
        self.size = size
        self.bandwidth_graph = bandwidth_graph
        self.latency_graph = latency_graph
        self.policy = policy

    def set_parallel_degree(self, parallel_degree):
        self.parallel_degree = parallel_degree

    def set_transmission_size(self, size):
        self.size = size

    def set_bandwidth_graph(self, graph):
        self.bandwidth_graph = graph
    
    def set_latency_graph(self, graph):
        self.latency_graph = graph

    def _get_local_rank0_list(self, ):
        local_rank0_list = []
        world_rank = 0
        visit = {}
        for ip in self.ip_table:
            if ip not in visit:
                local_rank0_list.append(world_rank)
                visit[ip] = True
            world_rank += 1
        return local_rank0_list

    def set_ip_info(self, ip_table):
        self.ip_table = ip_table
        self.local_rank0_list = self._get_local_rank0_list()

    def generate_strategy(self, prim):
        if self.policy == 'gurobi':
            if prim not in ['reduce', 'broadcast', 'alltoall']:
                print("prim not within the formulation scope.")
                return None
            else:
                solver = Solver()
                return solver.optimize(
                        prim, self.parallel_degree, self.size,
                        self.bandwidth_graph, self.latency_graph, 
                        self.strategy_file
                    )
        else:
            par_trees = ParTrees()
            return par_trees.optimize(self.ip_table, self.local_rank0_list, prim, 
                        self.parallel_degree, self.size,
                        self.bandwidth_graph, self.latency_graph, 
                        self.strategy_file
                    )
