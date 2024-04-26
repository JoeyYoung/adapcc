/*
    nvcc -L/home/xyzhao/openmpi/lib -I/home/xyzhao/openmpi/include -I./include -lmpi xxx cpp -o run
    will automatically search .h in current directory
*/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include "mpi.h"
#include "cuda_runtime.h"

#include <infiniband/verbs.h>

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

void getHostsAndLocalRank(int* localRank, int myRank, int nRanks, uint64_t* hostHashs){
    int localrank = 0;

    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) {
            break;
        }
        if (hostHashs[p] == hostHashs[myRank]) {
            localrank++; 
        }
    }
    *localRank = localrank;
}

int main(int argc, char *argv[]){
    int myRank, nRanks, localRank = 0;
    MPI_Init(&argc, &argv);
        
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    uint64_t hostHashs[nRanks];
    getHostsAndLocalRank(&localRank, myRank, nRanks, hostHashs);

    int offset = 2;
    int gpuId = (localRank + offset) % 4; // 为了使用rdma nic
    cudaSetDevice(gpuId);

    
    int size = 100*1024*1024; // 400MB
    float* buffer_host = (float*) malloc(sizeof(float) * size);
    for(int i = 0; i < size; i++)   buffer_host[i] = float(myRank)*float(myRank) + 1.0;

    float* localTensorDevBuffer;
    cudaMalloc(&localTensorDevBuffer, size * sizeof(float));
    cudaMemcpy(localTensorDevBuffer, buffer_host, size*sizeof(float), cudaMemcpyHostToDevice);

    float* recvTensorDevBuffer;
    cudaMalloc(&recvTensorDevBuffer, size * sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);

    clock_t start, end; 
    start = clock();
    // g16 gpu3 send flow
    if(gpuId == 2 && int(hostHashs[myRank])==187450375){
        MPI_Send(localTensorDevBuffer, size/10, MPI_FLOAT, 1, 3, MPI_COMM_WORLD); // tag 3
    }

    if(gpuId == 2 && int(hostHashs[myRank])==187450376){
        MPI_Recv(recvTensorDevBuffer, size/10, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    end = clock();
    printf("time=%f(ms)\n",((double)end-start)/CLOCKS_PER_SEC*1000);       

    MPI_Finalize();

    printf("[FLOW MPI Rank %d] Success \n", myRank);
    return 0;
}