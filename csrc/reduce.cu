/*
    Reduce Context
    - multiple parallel transmission contexts
*/

#include "control.h"
#include "trans.h"
#include "time.h"
#include <vector>
#include <iostream>
using namespace std;

static std::queue<workElement*> workQueueReduce[MAX_TRANS];
static std::queue<resultElement*> resultQueue[MAX_TRANS];

static MPIInfo* myMPIInfo;
static int numTrans;
static RankRoleInfo roles[MAX_TRANS][MAX_DEVICES];

static char* ipTable[MAX_DEVICES];
static vector<int> localMasterList;
static pthread_t pidsReduce[MAX_TRANS];

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

static void* reduceThreadFunc(void* args){
    cudaSetDevice(myMPIInfo->localRank);
    cudaDeviceEnablePeerAccess(myMPIInfo->localRank, 0);
    int tid = *(int*)args;
    cudaStream_t streamR;
    cudaStreamCreate(&streamR);

    RankRoleInfo role = roles[tid][myMPIInfo->myRank];
    float* recvDevBuffer;
    cudaMalloc(&recvDevBuffer, sizeof(float)*MAX_BUF_SIZE*role.precedents.size());
    
    sharedMemoryInfo shm_info;
    volatile shmStruct* shm;

    char shm_name[] = "shm_nameR_";
    shm_name[10] = '0' + tid;
    sharedMemoryCreate(shm_name, sizeof(*shm), &shm_info);
    shm = (volatile shmStruct *)shm_info.addr;
    cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->memHandle[myMPIInfo->myRank], recvDevBuffer
    );

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
    int shmID = shmEventRecordsCreate(&eventRecords, tid, 0, 
                    myMPIInfo->myRank, myMPIInfo->nRanks);
    
    globalInitBarrier(tid);
    printf("[Rank %d]global init done.\n", myMPIInfo->myRank);
    void* ptrs[myMPIInfo->nRanks];
    cudaEvent_t ipcEvents[myMPIInfo->nRanks][MAX_CHUNK_NUM];
    set<int>::iterator it;

    for(it=role.subsequents.begin(); it!=role.subsequents.end(); it++){
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
            cudaIpcOpenEventHandle(&p, 
                *(cudaIpcEventHandle_t *)&shm->eventHandle[i][cid]);
            ipcEvents[i][cid] = p;
        }
    }
    unordered_map<int, bool> checkCrossNodeSend;
    unordered_map<int, bool> checkCrossNodeRecv;
    unordered_map<int, int> checkPreSiblingIdx;

    for(it=role.subsequents.begin(); it!=role.subsequents.end(); it++){
        int dstRank = *it;
        if(strcmp(ipTable[myMPIInfo->myRank], ipTable[dstRank]) != 0){
            printf("[Rank %d]Reduce Thread: Cross send to %d\n", 
                    myMPIInfo->myRank, dstRank);
            checkCrossNodeSend[dstRank] = true;
        }
    }

    int preCount = 0;
    // cross node receive check
    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
        int srcRank = *it;
        if(strcmp(ipTable[myMPIInfo->myRank], ipTable[srcRank]) != 0){
            printf("[Rank %d]Reduce Thread: Cross receive from %d\n", 
                    myMPIInfo->myRank, srcRank);
            checkCrossNodeRecv[srcRank] = true;
        }
        preCount += 1;
        checkPreSiblingIdx[srcRank] = role.precedents.size()-preCount;
    }

    relayController* controller = (relayController*)malloc(sizeof(struct relayController_st));
    while(!exitSignal){
        if(workQueueReduce[tid].size() != 0){
            workElemBarrier(tid, eventRecords);

            workElement* elem = workQueueReduce[tid].front();
            resultElement* resElem = resultQueue[tid].front();

            int tranSize = elem->size/numTrans; 
            int tranChunkNum = sizeof(float)*tranSize/elem->chunkBytes; 
            int chunkFloatNum = elem->chunkBytes/sizeof(float); 
            int localBufOffset = tid*tranSize;

            printf("[Rank %d]fetch: address %p\n", myMPIInfo->myRank, elem->tensorBuf);
            float* localTensorDevBuffer = elem->tensorBuf;

            setRelayController(controller, roles[tid], myMPIInfo->myRank, 'r');
            set<int> recvSet = getActiveRecvs(roles[tid], myMPIInfo->myRank, 'r');
            if(role.subsequents.size() == 0){
                if(controller->hasKernel || !controller->hasRecv){  
                    resElem->resultDevBuf = localTensorDevBuffer+localBufOffset;
                }else{
                    if(recvSet.size() != 1) 
                        printf("[Rank %d]Conflict in root<->bcst control.\n", myMPIInfo->myRank);
                    resElem->resultDevBuf = recvDevBuffer +
                                            MAX_BUF_SIZE * checkPreSiblingIdx[*recvSet.begin()];
                }
            }else{
                resElem->resultDevBuf = localTensorDevBuffer+localBufOffset;
            }

            char signal = 'C';
            if(role.precedents.size() == 0 && role.subsequents.size() == 0)
                return 0;
            for(int cid = 0; cid < tranChunkNum; cid++){
                int globalChunkID = tid*tranChunkNum+cid;
                if(controller->hasRecv){
                    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
                        int srcRank = *it;
                        if(recvSet.count(srcRank) == 0)
                            continue;

                        int recvOffset = MAX_BUF_SIZE * checkPreSiblingIdx[srcRank]
                                            + cid * chunkFloatNum;
                        bool isCrossNode = (checkCrossNodeRecv.count(srcRank) != 0);
                        cudaRecv(
                            srcRank,
                            recvDevBuffer,
                            recvOffset,
                            chunkFloatNum,
                            &signal,
                            globalChunkID,
                            isCrossNode,
                            streamR,
                            ipcEvents[srcRank][globalChunkID],
                            eventRecords,
                            shmEventRecordsIdx(srcRank, globalChunkID)
                        );
                    }
                }

                if(controller->hasKernel){
                    int nBlocks = NUM_KERN_BLOCK;
                    int nThreads = NUM_KERN_THREADS;
                    int elemNum = role.precedents.size();

                    reduceSumKernel <<<nBlocks, nThreads, 0, streamR>>>(
                        recvDevBuffer, 
                        localTensorDevBuffer+localBufOffset+cid*chunkFloatNum,
                        cid, 
                        chunkFloatNum, 
                        MAX_BUF_SIZE, 
                        elemNum, 
                        controller->hasLocal
                    );
                    cudaStreamSynchronize(streamR);
                }

                if(controller->hasSend){
                    float* sendBuf = localTensorDevBuffer;
                    int sendOffset = localBufOffset+cid*chunkFloatNum;
                    if(controller->hasRecv && !controller->hasKernel){
                        if(recvSet.size() != 1) 
                            printf("[Rank %d]Conflict in send control.\n", myMPIInfo->myRank);
                        sendBuf = recvDevBuffer;
                        sendOffset = MAX_BUF_SIZE*checkPreSiblingIdx[*recvSet.begin()]
                                        + cid * chunkFloatNum;
                    }

                    for(it=role.subsequents.begin(); it!=role.subsequents.end(); it++){
                        int dstRank = *it;
                        void* ptr = ptrs[dstRank];
                        int startOffset = role.siblingIdx*MAX_BUF_SIZE*sizeof(float);            
                        bool isCrossNode = (checkCrossNodeSend.count(dstRank) != 0);
                        bool setRecord = (*it==*(role.subsequents.rbegin()) ? true: false);
                        cudaSend(
                            myMPIInfo->myRank, 
                            dstRank, 
                            myMPIInfo->localRank,
                            ptr,
                            startOffset+cid*elem->chunkBytes,
                            sendBuf,
                            sendOffset,
                            chunkFloatNum,
                            streamR,
                            &signal,
                            globalChunkID,
                            isCrossNode,
                            ipcEvents[myMPIInfo->myRank][globalChunkID],
                            eventRecords,
                            shmEventRecordsIdx(myMPIInfo->myRank, globalChunkID), 
                            setRecord
                        );
                    }
                }
            }
            resElem->status = true;
            printf("[Rank %d]work elem done.\n", myMPIInfo->myRank);
            workElemBarrierReb();
            workQueueReduce[tid].pop();
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

/* API */
namespace reduceContext{

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
        workQueueReduce[tid].push(elem);
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
        pthread_create(&pidsReduce[tid], NULL, 
            reduceThreadFunc, (void *)&(threadArgs[tid]));
    }
}

void clear(){
    void* status;
    for(int tid = 0; tid < numTrans; tid++){
        pthread_join(pidsReduce[tid], &status);
    }
}

void terminate(){
    exitSignal = true;
}

int getNumTrans(){
    return numTrans;
}

} // end namespace
