#ifndef SHM_IPC_H
#define SHM_IPC_H

#include "cuda_runtime.h"
#include "init.h"

#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#include <vector>
#include <sys/shm.h>

// define the ds stored in shm
typedef struct shmStruct_st {
    cudaIpcMemHandle_t memHandle[MAX_DEVICES];
    cudaIpcEventHandle_t eventHandle[MAX_DEVICES][MAX_CHUNK_NUM];
} shmStruct;

// define the meta info
typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);

/**
 * @brief used for the shm of events records
 * 
 * 	build the intra-server transmission signal，
 *  A->B, B wait events to know whether A has recored
 * 
 * @params: records  shms pointers
 * @params: tid  transmission context id
 * @params: type 0 for reduce (shmKey = 1000 + 100*tid), 1 for bcst (shmKey = 2000 + 100*tid)
 * 
 * @return: shm ID
 */
int shmEventRecordsCreate(bool ** records, int tid, int type, int myRank, int nRanks);


/**
 * @brief Detach eventRecords and remove shm segment with shm id
 */
void shmEventRecordsRemove(bool ** eventRecords, int shmID);


/**
 * @brief create eventRecords for a rank for a chunk
 * 
 * convert to 1d value: randId * MAX_CHUNK_NUM + chunkId
 */
int shmEventRecordsIdx(int rankId, int chunkId);

#endif // SHM_IPC_H
