/*
    Test for a simple tree, should support GPU in any server
            GPU0 --- GPU1
                      | 
                ------------
               GPU2       GPU3
    test per process per GPU, cuda IPC
    used for test whether actions of one GPUs can be implemented into one kernel

*/

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
    
    // create stream
    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaSetDevice(localRank);

    // init host data buffer value, tensor value
    int size = 32; // 4MB data
    int chunk_bytes = 128; 
    int chunk_num = size * sizeof(float) / chunk_bytes;
    int chunk_size = chunk_bytes / sizeof(float);

    float* localbuffer_host = (float*) malloc(sizeof(float) * size);
    for(int i = 0; i < size; i++)   localbuffer_host[i] = float(myRank)*float(myRank) + 1.0; // 5, 10 -> 2 -> 1

    float* sendbuffer_device, *recvbuffer_device;
    cudaMalloc(&sendbuffer_device, size * sizeof(float));
    
    // Gather all GPUs recv buffer pointers
    float* recvbuffer_pointers[nRanks];
    if(myRank == 1){
        cudaMalloc(&recvbuffer_device, 2 * size * sizeof(float));
        recvbuffer_pointers[myRank] = recvbuffer_device;
    }else{
        cudaMalloc(&recvbuffer_device, size * sizeof(float));
        recvbuffer_pointers[myRank] = recvbuffer_device;
    }

    // MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuffer_pointers, sizeof(float*), MPI_BYTE, MPI_COMM_WORLD);
    // printf("done mpi all gather\n");

    // enable all peers，with uva
    // for(int i=0; i<nRanks; i++){
    //     cudaSetDevice(i);
    //     for(int j=0; j<nRanks; j++){
    //         if(j == i) continue;
    //         cudaDeviceEnablePeerAccess(j, 0);
    //     }
    // }
    // printf("peer access enabled\n");
    
    // TODO, use ipc handle?
    if (myRank == 2){
        // send data to GPU 1
        cudaMemcpy(sendbuffer_device, localbuffer_host, size*sizeof(float), cudaMemcpyHostToDevice);

        cudaIpcMemHandle_t* handle;
        cudaIpcGetMemHandle(handle, sendbuffer_device);

        printCudaArray(sendbuffer_device, size);
    }else if(myRank == 1){
        sleep(10);
        // needs shared memory to store handle
        cudaIpcMemHandle_t* handle;
        void *ptr = NULL; // device pointer
        
        cudaIpcOpenMemHandle(&ptr, *handle, cudaIpcMemLazyEnablePeerAccess);

        cudaMemcpyPeer(recvbuffer_pointers[1], 1, sendbuffer_device, 2, size);
        printCudaArray(recvbuffer_pointers[1], size);
    }

    cudaFree(sendbuffer_device);
    cudaFree(recvbuffer_device);
    cudaFree(localbuffer_host);

    MPI_Finalize();

    printf("[CUDA Tree MPI Rank %d] Success \n", myRank);
    return 0;
}