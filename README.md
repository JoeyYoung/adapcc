# AdapCC: Making Collective Communication in Distributed Machine Learning Adaptive

AdapCC is a communication library that dynamically adapts to resource heterogeneity and network variability for optimized training performance. The main features are offered as follows:

- **Detecting.** adaptive to various resource allocations, by inferring physical configurations within each server.

- **Profiling.** adaptive to dynamic network changes, by coordinating workers to enable profiling on the fly.

- **Relay Control.** adaptive to computation stragglers, by allowing an arbitrary subset of workers to perform a collective. Non-active GPUs are controlled as relays for data transfers.

- **Fault Tolerance.** continued communication without being blocked (hang) by the straggler/faulty.

## Prerequisites
### Software Dependencies
**PyTorch**
- Python>=3.8
- PyTorch==1.13.0
- CUDA>=10.2
- GCC9.4
  
Download and compile the following libraries if you have not installed them:

**UCX==1.13.0**
```
wget https://github.com/openucx/ucx/releases/download/v1.13.0-rc1/ucx-1.13.0.tar.gz
tar xzf ucx-1.13.0.tar.gz
cd ucx-1.13.0
mkdir build
./contrib/configure-release --prefix=[BUILD_PATH] --with-cuda=[CUDA_PATH]
make -j8
make install
```
**OpenMPI==4.1.1**
```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
gunzip -c openmpi-4.1.1.tar.gz | tar xf -
cd openmpi-4.1.1
./configure --prefix=[BUILD_PATH] --with-cuda=[CUDA_PATH] --with-ucx=[UCX_PATH] --enable-mca-no-build=btl-uct
make all install
```
Add built paths to environment variables `PATH`, `LD_LIBRARY_PATH` and `MANPATH`, respectively.

### System Hardwares
Our testbed environment includes:
- Ubuntu 20.04 LTS
- NVIDIA A100 SXM4 40G GPUs with NVLink
- NVIDIA V100 SXM2 32G GPUs with NVLink
- Mellanox NIC 100Gbps / 50Gbps
- EPYC-7H12 CPU, PCIe 4.0 / Intel 6230 CPU, PCIe 3.0

## Install
Download the repo:

`git clone https://github.com/JoeyYoung/adapcc.git`

`cd adapcc` and edit `OMPI_LIB_DIR` and `OMPI_INC_DIR` to your own paths in the Makefile.

Run `make`. The compilation outputs a dynamic library `communicator.so`. 

## Usage
### Start with Launcher
Processes are managed by MPI. Use the script `launch_script.sh` to launch. 

The arguments include:
- `num_process:` &nbsp; the number of workers (processes) to start.
- `ips:` &nbsp; host ips of workers, following the format of mpirun.
- `master:` &nbsp; the ip of world rank 0, i.e., the master worker.
- `mpi_path:` &nbsp; execution path for mpirun.
- `net_device:` &nbsp; NIC interfaces for network communication.
- `exec_file:` &nbsp; the execution file name. You are required to provide this main file.
- `socket_port:` &nbsp; ports used for inter-process communication.
- `entry_point:` &nbsp; customize the value to enable the detection or profiling modules (if needed). 
- `logical_graph:` &nbsp; dump path of the logical graph. You can customize your server configuration as graph input, based on which profiling is performed. See examples in `'./topology'`.
- `strategy_file:` &nbsp; dump path of the communication topology strategies. Represented in xml format. You can customize your own strategy as input, based on which the data transfer follows. See examples in `'./strategy'`.
- `parallel_degree:` &nbsp; the number of parallel concurrent transmissions within one communication context.
- `profile_freq:` &nbsp; the frequency of profiling and graph construction, if enabled. 

### Primitive Example
Here are steps on how to run a communication operator, refer to `adapcc.py`:

1. Import library
   `import adapcc`.
2. Initialize
   `AdapCC.init(args, LOCAL_RANK, WORLD_RANK, WORLD_SIZE)`.
   It generates a communication strategy based on detection and profiling. A specific strategy could also be defined by users as illustrated in launcher.
3. Transmission context setup
   `AdapCC.setup(ALLREDUCE)`.
   Create communication resources and work queues for processing operators. 
4. Call primitives
   `AdapCC.communicator.all_reduce(tensor, size, chunk_bytes)`
5. Completed and reclaim resources
    `AdapCC.clear(ALLREDUCE)`. 

```python3
import adapcc
AdapCC.init(args, LOCAL_RANK, WORLD_RANK, WORLD_SIZE)
AdapCC.setup(ALLREDUCE)
...
AdapCC.communicator.all_reduce(tensor, size)
...
AdapCC.clear(ALLREDUCE)
```

You will obtain an output similar to [this](https://github.com/JoeyYoung/adapcc/blob/main/log/primitive).

**Note** 
1. The first op is always slow due to the [cache reason](https://forums.developer.nvidia.com/t/first-kernel-run-is-slower-than-succeeding/199915).
2. Adaptive Relay functionality is only supported in a training context.

### Training Example
`train_ddp.py` provides a training template. Use this for your models.

Compared to the former example, additionally, we register the hook `AdapCC.communicator.cuda_allreduce_hook` and call `AdapCC.reconstruct_topology` with a defined frequency to reconstruct communication topology.

Relay control is enabled by default and the output should be similar to [this](https://github.com/JoeyYoung/adapcc/blob/main/log/training).

## Contact
Raise issues or email xyzhao@cs.hku.hk for any questions.

## License
© Contributors Licensed under an Apache-2.0 license.
