#ifndef INIT_H
#define INIT_H

#include "mpi.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

/*
    system wide parameters, control flags
*/
// maximal workers number, increase for user scale
#define MAX_DEVICES 16
// maximal parallel degree, default 4
#define MAX_TRANS 8
// for pre ipc once, reuse the buffer
#define MAX_BUF_SIZE 400*1024*1024
// kernel related
#define NUM_KERN_BLOCK 512
#define NUM_KERN_THREADS 1024
#define MAX_CHUNK_NUM 512
// if torchrun
#define ELASTIC_LAUNCH false

/*
    Related to process initialization and rank info management
*/
typedef struct initMPIInfo_st{
    // drop host hashes, use ip table read from xml
    int myRank;
    int nRanks;
    int localRank;
    bool MPI_INIT;
} initMPIInfo;

initMPIInfo* getInitRanksInfo();
void setInitRanksInfo(int myRank, int nRanks, int localRank, bool MPI_INIT);

static void getHostName(char *hostname, int maxlen);
static uint64_t getHostHash(const char *string);
void getHostsAndLocalRank(int* localRank, int myRank, int nRanks, uint64_t* hostHashs);
void printCudaArray(float* device, int size);

#endif
