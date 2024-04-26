import os
import argparse
from multiprocessing import Process
import time

parser = argparse.ArgumentParser(description='env configurations')
parser.add_argument('--interval', type=int, default=5, help='send a bandwidth probe lasting -i seconds')
parser.add_argument('--duration', type=int, default=300, help='lasting the process for -d mins')
parser.add_argument('--log', type=str, default="./bandwidth.txt", help='file to store the results')
args = parser.parse_args()

probe_num = int(args.duration * 60 / args.interval)

log_file = open(args.log, "w")
for i in range(probe_num):
    cmd = os.popen("iperf -c 172.31.17.202 -i %d -t %d" % (args.interval, args.interval) )
    result = cmd.read()
    """
    ------------------------------------------------------------
    Client connecting to 172.31.17.202, TCP port 5001
    TCP window size:  325 KByte (default)
    ------------------------------------------------------------
    [  1] local 172.31.20.155 port 55146 connected with 172.31.17.202 port 5001
    [ ID] Interval       Transfer     Bandwidth
    [  1] 0.0000-5.0000 sec  5.55 GBytes  9.54 Gbits/sec
    [  1] 0.0000-5.0054 sec  5.55 GBytes  9.53 Gbits/sec
    """
    line = result.split("\n")[6]
    log_file.write(line)
    log_file.write("\n")

log_file.close()
