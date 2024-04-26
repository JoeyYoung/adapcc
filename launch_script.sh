# ps -ef | awk '{print $2,$9}' | grep rpc_server.py | awk '{print $1}' | xargs kill
# echo "terminating coordinator successfully"

# Check List
# (1) num-process: number of worker GPUs
# (2) ips: hostfile, process group
# (3) master: master worker, rank 0 ip
# (4) mpi-path: path to mpirun bin
# (5) net-device: NIC interfaces for ucx
# (6) exec-file: execution file
# The followings are required for execution file:
# (7) socket_port: port for threads/processes barrier, change if conflict
# (8) entry_point: the entry point of a workflow to start the job
# (9) logical_graph: logical file detected (if needed)
# (10) strategy_file: communication strategy file (if needed)

# We assume the intermediate files are stored with relavent path with a specific name
# logical graph should be given if start with profile 

python launcher.py \
    --num-process 4 \
    --ips 10.28.1.31:4 \
    --master 10.28.1.31 \
    --mpi-path /home/xyzhao/openmpi/bin/mpirun \
    --net-device mlx5_0:1 \
    --exec-file train_ddp.py \
    --socket_port 5000 \
    --entry_point -1 \
    --logical_graph ./topology/logical_graph_test.xml \
    --strategy_file ./strategy/strategy_test.xml \
    --parallel_degree 1 \
    --profile_freq 500  
