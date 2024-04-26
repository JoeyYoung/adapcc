'''
    @ Launcher:
    1. Pack mpirun and launch multiple processes
    2. Dump the ip table with ip for each rank (./topology/ip_table.txt)
    3. Dispatch the ip table
    4. Stored by the cudacom for init

    Note:
        The rank id starts from the master node

    Example: 
        see launch_script.sh
'''

import os
import argparse
from dispatcher import Dispatcher

parser = argparse.ArgumentParser()
parser.add_argument("--num-process", type=int, default="4")
parser.add_argument("--ips", type=str, default="10.28.1.31:4")
parser.add_argument("--master", type=str, default="10.28.1.31")
parser.add_argument("--mpi-path", type=str, default="/home/xyzhao/openmpi/bin/mpirun")
parser.add_argument("--net-device", type=str, default="mlx5_0:1")
parser.add_argument("--exec-file", type=str, default="mpirun_ddp.py")
parser.add_argument("--socket_port", type=str, default="5000")
parser.add_argument("--entry_point", type=int, default=-1, help="6:detect, 7:profile, other None")
parser.add_argument("--strategy_file", type=str, default="./strategy/strategy_test.xml")
parser.add_argument("--logical_graph", type=str, default="./topology/logical_graph_test.xml")
parser.add_argument("--parallel_degree", type=int, default=4)
parser.add_argument("--profile_freq", type=int, default=500)
args = parser.parse_args()

# patch launch command
launcher = "%s -np %d " % (
    args.mpi_path,
    args.num_process
)

hosts = "-H %s " % args.ips

flags_master = "-mca pml ucx -x UCX_NET_DEVICES=%s \
        --mca pml_base_verbose 10 --mca mtl_base_verbose 10 \
        -x MASTER_ADDR=%s " % (
    args.net_device,
    args.master
)

flags_port = "-x MASTER_PORT=1234 "
flags_path = "-x PATH "
flags_lib = "-x LD_LIBRARY_PATH "

# required fileds for execution files
command = "python %s %s %s %s %s %s %s" % (
    args.exec_file, 
    '--port=%s' % args.socket_port,
    '--entry_point=%d' % args.entry_point,
    '--strategy_file=%s' % args.strategy_file,
    '--logical_graph=%s' % args.logical_graph,
    '--parallel_degree=%d' % args.parallel_degree,
    '--profile_freq=%d' % args.profile_freq
)

# extract ip table
ips = []
sketch = open("./topology/ip_table.txt", "w")
# sketch file implying the rank id: ip
hosts_items = hosts.split(",")
for i in range(len(hosts_items)):
    host = hosts_items[i].strip()
    if i == 0:
        host = host.split(" ")[1].strip()
    ip = host.split(":")[0]
    rank_num = host.split(":")[1]
    for j in range(int(rank_num)):
        sketch.write(ip)
        sketch.write("\n")
    ips.append(ip)
sketch.close()

dispatcher = Dispatcher(ips)
work_dir = os.path.dirname(os.path.abspath(__file__))
dispatcher.dispatch_ip_table("./topology/ip_table.txt", "%s/topology" % work_dir)

run = launcher + hosts + flags_master + flags_port + flags_path + flags_lib + command
os.system(run)
