import os, sys
import xmltodict
import copy

class Node:
    def __init__(self, world_rank, ip, bandwidth, latency):
        self.world_rank = world_rank
        self.ip = ip
        self.bandwidth = bandwidth
        self.latency = latency
        self.bdp = self.bandwidth * self.latency
        self.left = None
        self.right = None

def construct_binary_tree(nodes):
    for i in range(len(nodes)):
        left_child_index = 2 * i + 1
        right_child_index = 2 * i + 2
        if left_child_index < len(nodes):
            nodes[i].left = nodes[left_child_index]
        if right_child_index < len(nodes):
            nodes[i].right = nodes[right_child_index]

    return nodes[0]

def levelOrder(root):
    if not root: return []
    res = []
    queue = []
    queue.append(root)

    while len(queue) > 0:
        num_layer = len(queue)
        layer = []
        for i in range(num_layer):
            node = queue.pop(0)
            layer.append(node.world_rank)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
            
        res.append(layer)
    
    return res

def dfs_intra_graph(idx, ip, config, gpu_list):
    if idx == len(gpu_list):
        return
    if idx == len(gpu_list)-1:
        config.append(
            {
                '@id': gpu_list[idx],
                '@ip': ip
            }
        )
        return
    
    config.append(
        {
            '@id': gpu_list[idx],
            '@ip': ip,
            'gpu': []
        }
    )
    
    dfs_intra_graph(idx+1, ip, config[0]['gpu'], gpu_list)

def traverse_tree_to_xml(root, config, gpus_group):
    if root.left == None and root.right == None \
            and len(gpus_group[root.world_rank]) <= 1:
        return
    
    ''' Split policy
        config['gpu'] = []
        for intra_g in gpus_group[root.world_rank]:
            if intra_g == root.world_rank:
                continue
            config['gpu'].append(
                {
                    '@id': intra_g,
                    '@ip': root.ip,
                }
            )
    '''
    
    ''' Chain policy
    '''
    config['gpu'] = []
    dfs_intra_graph(1, root.ip, config['gpu'], gpus_group[root.world_rank])

    if root.left == None and root.right == None:
        return
    if root.left != None:
        config['gpu'].append(
            {
                '@id': root.left.world_rank,
                '@ip': root.left.ip,
            }
        )        
        traverse_tree_to_xml(root.left, config['gpu'][-1], gpus_group)
    if root.right != None:
        config['gpu'].append(
            {
                '@id': root.right.world_rank,
                '@ip': root.right.ip,
            }
        )
        traverse_tree_to_xml(root.right, config['gpu'][-1], gpus_group)


class ParTrees:
    def __init__(self):
        pass
    
    def optimize(self, ip_table, local_rank0_list, prim, parallel_degree, 
                    transmission_size, bandwidth_graph, latency_graph, strategy_file):
        node_list_ = []
        gpus_group = {} # {localrank0: [0, 1, 2, 3]}
        default_chunk = 4*1024*1024

        for rank in local_rank0_list:
            gpus_group[rank] = []
            idx = rank
            while ((idx < len(ip_table) and ip_table[idx] == ip_table[rank])):
                gpus_group[rank].append(idx)
                idx += 1    
            node_list_.append(
                Node(rank, ip_table[rank], 
                    bandwidth_graph[rank][(rank+len(gpus_group[rank])) % len(ip_table)],
                    latency_graph[rank][(rank+len(gpus_group[rank])) % len(ip_table)]
                )
            )

        node_list = sorted(node_list_, key=lambda obj: obj.bdp, reverse=True)
        tree_list = []
        parallel_degree = min(len(local_rank0_list), parallel_degree)
        for tran in range(parallel_degree):
            if tran > 0: node_list = node_list[1:] + [node_list[0]]
            inter_node_root = construct_binary_tree(copy.deepcopy(node_list))
            tree_list.append(inter_node_root)
        
        strategy_graph = {'trees': {'root': []}}
        for root in tree_list:
            root_config = {}
            root_config['@id'] = root.world_rank
            root_config['@ip'] = root.ip
            traverse_tree_to_xml(root, root_config, gpus_group)
            strategy_graph['trees']['root'].append(copy.deepcopy(root_config))
            
        with open(strategy_file, 'w') as f:
            f.write(xmltodict.unparse(strategy_graph))

        return default_chunk
        