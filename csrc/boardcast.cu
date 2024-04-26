/*
    Broadcast Context
    - Tree based for reduced traffic volume
    - multiple parallel transmission contexts
*/

#include "control.h"
#include "trans.h"
#include "time.h"
#include <vector>
#include <iostream>
using namespace std;

static std::queue<workElement*> workQueueBcst[MAX_TRANS];
static std::queue<resultElement*> resultQueue[MAX_TRANS];
static MPIInfo* myMPIInfo;
static int numTrans;
static RankRoleInfo roles[MAX_TRANS][MAX_DEVICES];

static char* ipTable[MAX_DEVICES];
static vector<int> localMasterList;
static pthread_t pidsBroad[MAX_TRANS];
static int initCount;
static pthread_mutex_t initCountMutex;
static bool initDone;

static int workCount;
static pthread_mutex_t workCountMutex;
static bool workDone;

static socketFds* globalIpcFds;
static socketFds* workIpcFds;

static bool exitSignal;

static void treeDFS(XMLElement* node, int parent, int tid){
    int id = atoi(node->Attribute("id"));
    char* ip = (char*)node->Attribute("ip");
    ipTable[id] = ip;

    roles[tid][id].siblingIdx = 0;
    XMLElement* tmp = node->NextSiblingElement("gpu");
    while(tmp != NULL){
        tmp = tmp->NextSiblingElement("gpu");
        roles[tid][id].siblingIdx += 1;
    }
    if(parent != -1){
        roles[tid][id].subsequents.insert(parent);
    }
    XMLElement* child = node->FirstChildElement("gpu");
    while(child != NULL){
        int childId = atoi(child->Attribute("id"));
        roles[tid][id].precedents.insert(childId);
        child = child->NextSiblingElement("gpu");
    }

    if(node->FirstChildElement("gpu") == NULL)
        return;
    
    child = node->FirstChildElement("gpu");
    while(child != NULL){
        treeDFS(child, id, tid);
        child = child->NextSiblingElement();
    }
}

static void getStrategyFromXML(XMLElement* trees){
    XMLElement *root = trees->FirstChildElement("root");
    int count = 0;
    while(root != NULL){
        treeDFS(root, -1, count);
        count += 1;
        root = root->NextSiblingElement("root");
    }
    numTrans = count;

    char temp_ip[20];
    localMasterList.clear();
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(strcmp(ipTable[i], temp_ip) != 0){
            strcpy(temp_ip, ipTable[i]);
            localMasterList.push_back(i);
        }
    }
}

static void globalBarrier(){
    char initBarrier = 'I';
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcSend(globalIpcFds->sendFd[i], &initBarrier, 1);
    }
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcRecv(globalIpcFds->recvFd[i], &initBarrier, 1);
    }
}

static void globalInitBarrier(int tid){
    char initBarrier = 'I';
    pthread_mutex_lock(&initCountMutex);
    initCount += 1;
    pthread_mutex_unlock(&initCountMutex);
    
    if(tid != 0)
        while(!initDone);
    else{
        while(initCount == 0 || initCount%numTrans!=0);
        for(int i = 0; i < myMPIInfo->nRanks; i++){
            if(i == myMPIInfo->myRank) continue;
            socketIpcSend(globalIpcFds->sendFd[i], &initBarrier, 1);
        }
        for(int i = 0; i < myMPIInfo->nRanks; i++){
            if(i == myMPIInfo->myRank) continue;
            socketIpcRecv(globalIpcFds->recvFd[i], &initBarrier, 1);
        }
        initDone = true;
    }
}

static void workElemBarrier(int tid, bool* records){
    char workBarrier = 'W';
    pthread_mutex_lock(&workCountMutex);
    workCount += 1;
    pthread_mutex_unlock(&workCountMutex);

    if(tid != 0){
        while(!workDone);
    }
    else{
        while(workCount==0 || workCount%(numTrans)!=0);
        for(int i = 0; i < myMPIInfo->nRanks; i++){
            if(i == myMPIInfo->myRank) continue;
            socketIpcSend(workIpcFds->sendFd[i], &workBarrier, 1);
        }
        for(int i = 0; i < myMPIInfo->nRanks; i++){
            if(i == myMPIInfo->myRank) continue;
            socketIpcRecv(workIpcFds->recvFd[i], &workBarrier, 1);
        }
        workDone = true;
    }

    if(myMPIInfo->localRank == 0)
        memset(records, 0, myMPIInfo->nRanks*MAX_CHUNK_NUM);
    globalInitBarrier(tid);
}

static void workElemBarrierReb(){
    workDone = false;
}

static void* broadThreadFunc(void* args){
    cudaSetDevice(myMPIInfo->localRank);
    cudaDeviceEnablePeerAccess(myMPIInfo->localRank, 0);

    int tid = *(int*)args;
    cudaStream_t streamB;
    cudaStreamCreate(&streamB);
    RankRoleInfo role = roles[tid][myMPIInfo->myRank];

    float* recvDevBuffer;
    cudaMalloc(&recvDevBuffer, sizeof(float)*MAX_BUF_SIZE);
    
    // shared memory related
    sharedMemoryInfo shm_info;
    volatile shmStruct* shm;
    char shm_name[] = "shm_nameB_";
    shm_name[10] = '0' + tid;
    sharedMemoryCreate(shm_name, sizeof(*shm), &shm_info);
    shm = (volatile shmStruct *)shm_info.addr;
    cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->memHandle[myMPIInfo->myRank], recvDevBuffer
    );

    // ipc events, rank:global cid
    for(int i=0; i<myMPIInfo->nRanks; i++){
        for(int j=0; j<MAX_CHUNK_NUM; j++){
            cudaEvent_t event;
            cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess);
            cudaIpcGetEventHandle(
                (cudaIpcEventHandle_t *)&shm->eventHandle[i][j], event
            );
        }
    }
    bool* eventRecords;
    int shmID = shmEventRecordsCreate(
                &eventRecords, tid, 1, myMPIInfo->myRank, myMPIInfo->nRanks);

    globalInitBarrier(tid);

    void* ptrs[myMPIInfo->nRanks];
    cudaEvent_t ipcEvents[myMPIInfo->nRanks][MAX_CHUNK_NUM];

    set<int>::iterator it;
    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
        int dstRank = *it;
        void* p;
        cudaIpcOpenMemHandle(
            &p, *(cudaIpcMemHandle_t *)&shm->memHandle[dstRank], 
            cudaIpcMemLazyEnablePeerAccess
        );
        ptrs[dstRank] = p;
    }

    // pre-load event ipc
    for(int i=0; i<myMPIInfo->nRanks; i++){
        for(int cid=0; cid<MAX_CHUNK_NUM; cid++){
            cudaEvent_t p;
            cudaIpcOpenEventHandle(
                &p, *(cudaIpcEventHandle_t *)&shm->eventHandle[i][cid]
            );
            ipcEvents[i][cid] = p;
        }
    }

    // topology management
    unordered_map<int, bool> checkCrossNodeSend;
    unordered_map<int, bool> checkCrossNodeRecv;

    // cross node send check 
    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
        int dstRank = *it;
        if(strcmp(ipTable[myMPIInfo->myRank], ipTable[dstRank]) != 0){
            checkCrossNodeSend[dstRank] = true;
        }
    }

    // cross node receive check
    for(it=role.subsequents.begin(); it!=role.subsequents.end(); it++){
        int srcRank = *it;
        if(strcmp(ipTable[myMPIInfo->myRank], ipTable[srcRank]) != 0){
            checkCrossNodeRecv[srcRank] = true;
        }
    }

    char signal = 'C';
    while(!exitSignal){
        if(workQueueBcst[tid].size() != 0){
            workElemBarrier(tid, eventRecords);
            
            workElement* elem = workQueueBcst[tid].front();
            resultElement* resElem = resultQueue[tid].front();
            int tranSize = elem->size/numTrans;
            int tranChunkNum = sizeof(float)*tranSize/elem->chunkBytes; 
            int chunkFloatNum = elem->chunkBytes/sizeof(float);
            
            int localBufOffset = tid*tranSize;
            float* localTensorDevBuffer = elem->tensorBuf;
            if(role.subsequents.size() == 0){
                resElem->resultDevBuf = localTensorDevBuffer+localBufOffset;
            }else{
                resElem->resultDevBuf = recvDevBuffer;
            }

            if(role.precedents.size() == 0 && role.subsequents.size() == 0)
                return 0;
            for(int cid = 0; cid < tranChunkNum; cid++){
                int globalChunkID = tid*tranChunkNum+cid;
                for(it=role.subsequents.begin(); it!=role.subsequents.end(); it++){
                    int srcRank = *it;
                    bool isCrossNode = (checkCrossNodeRecv.count(srcRank) != 0);
                    cudaRecv(
                        srcRank,
                        recvDevBuffer,
                        cid*chunkFloatNum,
                        chunkFloatNum,
                        &signal,
                        globalChunkID,
                        isCrossNode,
                        streamB,
                        ipcEvents[srcRank][globalChunkID],
                        eventRecords,
                        shmEventRecordsIdx(srcRank, globalChunkID)
                    );
                }

                float* sendBuf = recvDevBuffer;
                int sendOffset = cid*chunkFloatNum;
                if(role.subsequents.size() == 0){
                    sendBuf = localTensorDevBuffer;
                    sendOffset = localBufOffset + cid*chunkFloatNum;
                }
                for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
                    int dstRank = *it;
                    void* ptr = ptrs[dstRank];      
                    bool isCrossNode = (checkCrossNodeSend.count(dstRank) != 0);
                    bool setRecord = (*it==*(role.subsequents.rbegin()) ? true: false);
                    cudaSend(
                        myMPIInfo->myRank, dstRank, myMPIInfo->localRank,
                        ptr,
                        cid*elem->chunkBytes,
                        sendBuf,
                        sendOffset,
                        chunkFloatNum,
                        streamB,
                        &signal,
                        globalChunkID,
                        isCrossNode,
                        ipcEvents[myMPIInfo->myRank][globalChunkID],
                        eventRecords,
                        shmEventRecordsIdx(myMPIInfo->myRank, globalChunkID), 
                        setRecord
                    );
                }
            } // end of chunks

        DEBUG:
            resElem->status = true;
            workElemBarrierReb();
            workQueueBcst[tid].pop();
        }
    }

    sharedMemoryClose(&shm_info);
    shmEventRecordsRemove(&eventRecords, shmID);
    cudaFree(recvDevBuffer);
    return 0;
}

static bool checkResQueue(){
    for(int tid = 0; tid < numTrans; tid++){
        if(!resultQueue[tid].front()->status)
            return false;
    }
    return true;
}


namespace boardcastContext{

void getMPIInfo(MPIInfo* info){
    myMPIInfo = info;
}

float** processWorkElem(workElement* elem){
    initDone = false;
    globalBarrier();
    printf("[Rank %d]enqueue: %p\n", myMPIInfo->myRank, elem->tensorBuf);

    struct timespec start = {0, 0};
    struct timespec end = {0, 0};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    resultElement* res[numTrans];
    for(int tid = 0; tid < numTrans; tid++){
        res[tid] = (struct resultElement_st*)malloc(sizeof(struct resultElement_st));
        resultQueue[tid].push(res[tid]);
    }

    for(int tid = 0; tid < numTrans; tid++){
        workQueueBcst[tid].push(elem);
    }

    while(!checkResQueue());
    
    float** returnBufs = (float**)malloc(numTrans * sizeof(float*));
    for(int tid = 0; tid < numTrans; tid++){
        returnBufs[tid] = resultQueue[tid].front()->resultDevBuf;
        resultQueue[tid].pop();
        free(res[tid]);
    }

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    double duration = (end.tv_sec - start.tv_sec) * 1000 
                        + (end.tv_nsec - start.tv_nsec) / 1000000;

    globalBarrier();
    return returnBufs;
}

void bootstrapFromXMl(char* filename){
    XMLDocument xml;
    xml.LoadFile(filename);
    XMLElement *trees = xml.RootElement();
    getStrategyFromXML(trees);

    socketFds** channels = buildIpcChannels(ipTable, 
                            myMPIInfo->myRank, myMPIInfo->nRanks);
    globalIpcFds = channels[0];
    workIpcFds = channels[1];

    int threadArgs[numTrans];
    for(int tid = 0; tid < numTrans; tid ++){
        threadArgs[tid] = tid;
        pthread_create(&pidsBroad[tid], NULL, 
                        broadThreadFunc, (void *)&(threadArgs[tid]));
    }
}

void clear(){
    void* status;
    for(int tid = 0; tid < numTrans; tid++){
        pthread_join(pidsBroad[tid], &status);
    }
}

void terminate(){
    exitSignal = true;
}

int getNumTrans(){
    return numTrans;
}

} // end namespace
