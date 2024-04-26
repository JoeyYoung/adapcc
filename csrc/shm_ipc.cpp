#include "shm_ipc.h"
#include <cstdlib>
#include <string>

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
    int status = 0;

    info->size = sz;

    info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (info->shmFd < 0) {
      	return errno;
    }

    status = ftruncate(info->shmFd, sz);
    if (status != 0) {
      	return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
      	return errno;
    }

    return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
    info->size = sz;

	info->shmFd = shm_open(name, O_RDWR, 0777);
	if (info->shmFd < 0) {
		return errno;
	}

	info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, info->shmFd, 0);
	if (info->addr == NULL) {
		return errno;
	}

	return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info) {
  if (info->addr) {
      munmap(info->addr, info->size);
  }
  if (info->shmFd) {
      close(info->shmFd);
  }
}

int shmEventRecordsCreate(bool** records, int tid, int type, 
                            int myRank, int nRanks){
    int shmID;
	key_t shmKey;

	if(type == 0){
		shmKey = 1000 + 100*tid;
	}else if(type == 1){
		shmKey = 2000 + 100*tid;
	}else{
		printf("[Rank %d]Wrong type to create shm\n", myRank);
	}
    
    shmID = shmget(shmKey, sizeof(int) * nRanks * MAX_CHUNK_NUM, 
                        IPC_CREAT | 0777);
    if(shmID < 0){
        // for case of error does not return a valid shmid
        printf("[Rank %d]Error getting shared memory\n", myRank);
        exit(EXIT_FAILURE);
    }else{
        printf("[Rank %d]Open shared memory %d for event records.\n", myRank, shmID);
    }

    *records = (bool *)shmat(shmID, NULL, 0); // attach memory

	return shmID;
}


void shmEventRecordsRemove(bool ** eventRecords, int shmID){
	// detech the ptr, wont delete the shared memory
    if(shmdt((void *)(*eventRecords)) == -1){
        printf("shmdt failed\n");
        exit(EXIT_FAILURE);
    }
    // remove shm with id
    shmctl(shmID, IPC_RMID, 0);
}


int shmEventRecordsIdx(int randId, int chunkId){
	return (randId * MAX_CHUNK_NUM + chunkId);
}
