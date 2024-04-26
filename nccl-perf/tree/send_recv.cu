/*
    Test for a simple tree, should support GPU in any server
        GPU0 ------ GPU1
                      | 
                ------------
               GPU2       GPU3
    test for the latency of nccl send and recv

*/

#include "nccl.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void printCudaArray(float* device, int size){
    float* host = (float*)malloc(sizeof(float)*size);
    cudaMemcpy(host, device, size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++)   printf("%f ", host[i]);
    free(host);
}

__global__ void reduceKernel(float* a, float* b, float* c, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]){
    int myRank, nRanks, localRank = 0;

    // initializing MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // obtain local rank
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) {
            break;
        }
        if (hostHashs[p] == hostHashs[myRank]) {
            localRank++;    // get local rank index
        }
    }

    // nccl id
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) {
        ncclGetUniqueId(&id);   // ncclInit: initEnv() initNet()
    }
    MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);


    // cuda part: allocate mem
    int size = 250 * 1024;  // 32 * 4 bytes = 128 bytes
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    cudaSetDevice(localRank);
    cudaStreamCreate(&s);

    // nccl init
    ncclCommInitRank(&comm, nRanks, id, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    clock_t start, end; 
    start = clock();
    if(myRank == 1){
        cudaMalloc(&sendbuff, size * sizeof(float));
        ncclSend(sendbuff, size, ncclFloat, 0, comm, s);
        cudaStreamSynchronize(s);
    }else if (myRank == 0){
        cudaMalloc(&recvbuff, size * sizeof(float));
        ncclRecv(recvbuff, size, ncclFloat, 1, comm, s);
        cudaStreamSynchronize(s);
    }
    else
        printf("only use two GPUs\n");

    end = clock();
    printf("time=%f(ms)\n",((double)end-start)/CLOCKS_PER_SEC*1000);       

    // free device buffers, finalizing NCCL, MPI
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    ncclCommDestroy(comm);
    MPI_Finalize();

    printf("[P2P Rank %d] Success \n", myRank);
    return 0;
}