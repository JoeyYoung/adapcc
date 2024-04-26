import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=4)
parser.add_argument("--ips", type=str, default="127.0.0.1:4")
parser.add_argument("--master", type=str, default="127.0.0.1")
parser.add_argument("--port", type=str, default="1234")
parser.add_argument("--cmd", type=str, default="echo HELLO")
args = parser.parse_args()

mpi_cmd = "/home/xyzhao/openmpi/bin/mpirun -np %d -H %s " % (args.num, args.ips)
flags_master = "-x MASTER_ADDR=%s " % args.master
flags_port = "-x MASTER_PORT=%s " % args.port
flags_path = "-x PATH "
flags_lib = "-x LD_LIBRARY_PATH "
flags = flags_master + flags_port + flags_path + flags_lib
exec_cmd = args.cmd

os.system(mpi_cmd + flags + exec_cmd)
