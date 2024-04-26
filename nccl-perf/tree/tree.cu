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
    int size = 32;
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    cudaSetDevice(localRank);
    cudaStreamCreate(&s);

    // nccl init
    ncclCommInitRank(&comm, nRanks, id, myRank);
    
    clock_t start,end;

    start = clock();
    // set reduce tree
    if (myRank == 2 or myRank == 3){
        // send data to GPU 1
        int dst = 1;
        cudaMalloc(&sendbuff, size * sizeof(float)); // 4 bytes unit size
        
        // set init value in host mem and copy to device
        float* sendbuff_host = (float*) malloc(sizeof(float) * size);
        for(int i = 0; i < size; i++)   sendbuff_host[i] = float(myRank);
        cudaMemcpy(sendbuff, sendbuff_host, size*sizeof(float), cudaMemcpyHostToDevice);
        free(sendbuff_host);
        
        ncclSend(sendbuff, size, ncclFloat, dst, comm, s);

        cudaStreamSynchronize(s);
    }else if(myRank == 1){
        // recv data from GPU 2 and GPU 3
        int dst = 0;
        int peer2 = 2;
        int peer3 = 3;
        cudaMalloc(&recvbuff, 2 * size * sizeof(float));
        cudaMalloc(&sendbuff, size * sizeof(float));
        
        ncclGroupStart();
        ncclRecv(recvbuff, size, ncclFloat, peer2, comm, s);
        ncclRecv(recvbuff + size, size, ncclFloat, peer3, comm, s);
        ncclGroupEnd();

        // reduce operation, set send buffer, wait for nccl recv?
        int nblocks = 2;
        reduceKernel <<<nblocks, size/nblocks>>> (recvbuff, recvbuff + size, sendbuff, size);
        cudaDeviceSynchronize();
        
        ncclSend(sendbuff, size, ncclFloat, dst, comm, s);

        cudaStreamSynchronize(s);
    }else if(myRank == 0){
        // recv data from GPU 1
        int src = 1;
        cudaMalloc(&recvbuff, size * sizeof(float));
        sleep(3);
        ncclRecv(recvbuff, size, ncclFloat, src, comm, s);
        
        cudaStreamSynchronize(s);
    }
    end = clock();
    printf("time=%f(ms)\n",(double)(end-start)/CLOCKS_PER_SEC * 1000);  

    // free device buffers, finalizing NCCL, MPI
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    ncclCommDestroy(comm);
    MPI_Finalize();

    printf("[Tree MPI Rank %d] Success \n", myRank);
    return 0;
}
