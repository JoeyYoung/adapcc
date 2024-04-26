/*
    This file contains implementations for cuda and mpi checking
*/

#include "init.h"
#include "mpi-ext.h"

// use to address multiple mpi init conflications
static initMPIInfo* initRanksInfo = (initMPIInfo*)malloc(sizeof(struct initMPIInfo_st));

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

void getHostsAndLocalRank(int* localRank, int myRank, int nRanks, 
                            uint64_t* hostHashs){
    int localrank = 0;

    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
        hostHashs, sizeof(uint64_t), 
        MPI_BYTE, MPI_COMM_WORLD
    );
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

initMPIInfo* getInitRanksInfo(){
    return initRanksInfo;
}

void setInitRanksInfo(int myRank, int nRanks, int localRank, bool MPI_INIT){
    initRanksInfo->myRank = myRank;
    initRanksInfo->nRanks = nRanks;
    initRanksInfo->localRank = localRank;
    initRanksInfo->MPI_INIT = MPI_INIT;
}

void printCudaArray(float* device, int size){
    float* host = (float*)malloc(sizeof(float)*size);
    cudaMemcpy(host, device, size * sizeof(float), 
                cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++)
        printf("%f ", host[i]);
    printf("\n");
    free(host);
}

void checkCudaAwareMPI(){
    printf("Compile time check:\n");
    #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf("This MPI library has CUDA-aware support.\n");
    #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
        printf("This MPI library does not have CUDA-aware support.\n");
    #else
        printf("This MPI library cannot determine if there is CUDA-aware support.\n");
    #endif /* MPIX_CUDA_AWARE_SUPPORT */
    
        printf("Run time check:\n");
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
            printf("This MPI library has CUDA-aware support.\n");
        } else {
            printf("This MPI library does not have CUDA-aware support.\n");
        }
    #else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        printf("This MPI library cannot determine if there is CUDA-aware support.\n");
    #endif /* MPIX_CUDA_AWARE_SUPPORT */
}
