#ifndef TRANS_H
#define TRANS_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <set>
#include <queue>
#include <pthread.h>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "cuda_runtime.h"
#include "tinyxml2.h"
#include "init.h"
#include "shm_ipc.h"
extern "C"{
#include "setup_ib.h"
}
    
using namespace std;
using namespace tinyxml2;

// Start with int 0
enum Primitive{
    ALLREDUCE,
    REDUCE,
    BOARDCAST,
    ALLGATHER, 
    ALLTOALL,
    REDUCESCATTER,
    DETECT,
    PROFILE
};

typedef struct MPIInfo_st{
    // drop host hashes, use ip table read from xml
    int myRank;
    int nRanks;
    int localRank;
} MPIInfo;

typedef struct RankRoleInfo_st{
    // role: upstream ranks
    set<int> precedents; 
    // role: downstream ranks
    set<int> subsequents;   
    // used for the case when multiple ranks sent to the dst
    // offset of the recv buffer
    int siblingIdx;
}RankRoleInfo;

// work element ds in the work queue
typedef struct workElement_st{
    float* tensorBuf;
    int size;
    int chunkBytes;
}workElement;

// result element in the result queue
typedef struct resultElement_st{
    bool status;
    float* resultDevBuf;
}resultElement;

// socket fds for commu with other ranks
typedef struct socketFds_st{
    int recvFd[MAX_DEVICES];
    int sendFd[MAX_DEVICES];
}socketFds;


// general transmission functions

/**
 * @brief aggregation kernel, default sum
 * 
 * @params: a  recv device buffer
 * @params: c  local device buffer
 * @params: cid  current chunk id
 * @params: size  float number of current chunk
 * @params: length  MAX_BUF_SIZE
 * @params: elnum  number of precedents
 * @params: active  status of current rank
 */
__global__ void reduceSumKernel(float* a, float* c, int cid, int size, 
                                int length, int elnum, int active);

__global__ void reduceAvgKernel(float* a, float* c, int cid, int size, 
                                int length, int elnum, int active);

__global__ void reduceMaxKernel(float* a, float* c, int cid, int size, 
                                int length, int elnum, int active);

/**
 * @brief general p2p sender function
 * 
 * @params: srcRank     sender world rank
 * @params: dstRank     receiver world rank
 * @params: srcDevice   sender local rank
 * @params: recvBuf     receiver ipc buffer
 * @params: recvOffset  receiver buffer offset
 * @params: localBuf    sender buffer
 * @params: localOffset sender buffer offset
 * @params: floatNum    total number of floats
 * @params: stream      transmission stream
 * @params: signal      mpi signal
 * @params: mpiTag      mpi tag, AKA global chunk id in current tensor
 * @params: isCrossNode whether inter-node transmission
 * @params: event       event corresponsing to the chunk
 * @params: records     whether have records for events, shm for other ranks in the same thread
 * @params: idx         rank id * MAX_CHUNK_NUM + global chunk id
 * @params: setRecord   each rank has one send record flag, but multiple receiver. trigger once
 */
void cudaSend(int srcRank, int dstRank, int srcDevice, void* recvBuf, int recvOffset, float* localBuf, int localOffset,
              int floatNum, cudaStream_t stream, char* signal, int mpiTag, bool isCrossNode,
              cudaEvent_t event, bool* records, int idx, bool setRecord);


/**
 * @brief general p2p receiver function
 * 
 * @params: srcRank     sender rank id
 * @params: recvBuf     receiver buffer 
 * @params: recvOffset  receiver buffer offset
 * @params: floatNum    number of floats
 * @params: signal      mpi signal
 * @params: mpiTag      mpi tag AKA global chunk id in current tensor
 * @params: isCrossNode whether inter-node transmission
 * @params: stream      transmission stream
 * @params: event       event corrsponding to the chunk
 * @params: records     whether have records for events
 * @params: idx         rank id * MAX_CHUNK_NUM + global chunk id
 */
void cudaRecv(int srcRank, float* recvBuf, int recvOffset, int floatNum,
              char* signal, int mpiTag, bool isCrossNode, cudaStream_t stream,
              cudaEvent_t event, bool* records, int idx);


/**
 * @brief create recv socket on one rank
 * 
 * as a server, create fd for other ranks in ip:port, listen & accept
 */
int createRecvSocket(int myRank, char* ip, int port);


/**
 * @brief create send socket on one rank
 * 
 * as a client, create fd for other ranks, connect
 */
int createSendSocket(int myRank, char* ip, int port);


/**
 * @brief generate consensus ports
 * 
 * @params: serverRank  as server rank accept / recv
 * @params: clientRank  as client rank connect / send
 * @params: type 'w' / 'g', for the barrier type, global or work elems
 */
int generatePort(int serverRank, int clientRank, char type);


/**
 * @brief build recv fds and send fds for socket ipc
 * 
 * ranks build sockets channels following three steps:
 * 1. ranks build recv fds
 * 2. select rank i as server in order, rank j as client, build j-->i channel
 * 3. j accpet and update recv fds, i create send fds and connect to j
 * 
 * @params: ipTable
 */
socketFds** buildIpcChannels(char** ipTable, int myRank, int nRanks);


/**
 * @brief for barrier ipc
 * 
 * wrap socket send, non-blocking operator
 * 
 * @params: fd
 * @params: signal  useless, for identification
 * @params: size    bytes
 */
void socketIpcSend(int fd, char* buf, int size);


/**
 * @brief for barrier ipc
 * 
 * wrap socket recv, blocking operator
 * 
 * @params: fd
 * @params: signal  useless, for identification
 * @params: size    bytes
 */
void socketIpcRecv(int fd, char* buf, int size);


/**
 * @brief for socket creation
 * 
 * global port = port
 * worker port = port + 1000
 * 
 * @params: port
 */
void attachSocketPort(int port);


/* 
    Primitive contexts
*/
namespace allreduceContext{
    void getMPIInfo(MPIInfo* info);
    void bootstrapFromXMl(char* filename);
    float** processWorkElem(workElement* elem);
    void clear();
    void terminate();
    int getNumTrans();
}

namespace reduceContext{
    void getMPIInfo(MPIInfo* info);
    void bootstrapFromXMl(char* filename);
    float** processWorkElem(workElement* elem);
    void clear();
    void terminate();
    int getNumTrans();
}

namespace boardcastContext{
    void getMPIInfo(MPIInfo* info);
    void bootstrapFromXMl(char* filename);
    float** processWorkElem(workElement* elem);
    void clear();
    void terminate();
    int getNumTrans();
}

/* blocking ops */
namespace detectContext{
    void getMPIInfo(MPIInfo* info);
    void bootstrapFromSketch(char* filename);
}

namespace profileContext{
    void getMPIInfo(MPIInfo* info);
    void bootstrapFromLogicGraph(char* filename);
}

#endif
