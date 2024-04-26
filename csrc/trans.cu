/*
    implementations for data transmission
*/

#include "trans.h"

static int GLOBAL_SOCKET_START_PORT; 
static int WORK_SOCKET_START_PORT;

__global__ void reduceSumKernel(float* a, float* c, int cid, int size, 
                                    int length, int elnum, int active) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < size; i += blockDim.x * gridDim.x){
        float tmp = 0; 
        for (int j = 0; j < elnum; j++){
            int offset = (cid*size + length*j);
            tmp += *(a+offset+i);
        }
        if(active)
            c[i] = tmp + c[i];
        else
            c[i] = tmp;
    }
}

__global__ void reduceAvgKernel(float* a, float* c, int cid, int size, 
                                    int length, int elnum, int active) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < size; i += blockDim.x * gridDim.x){
        float tmp = 0; 
        for (int j = 0; j < elnum; j++){
            int offset = (cid*size + length*j);
            tmp += *(a+offset+i);
        }
        if(active)
            c[i] = (tmp + c[i]) / (elnum + 1);
        else
            c[i] = tmp / elnum;
    }
}

__global__ void reduceMaxKernel(float* a, float* c, int cid, int size, 
                                    int length, int elnum, int active) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < size; i += blockDim.x * gridDim.x){
        float max = 0; 
        for (int j = 0; j < elnum; j++){
            int offset = (cid*size + length*j);
            if(*(a+offset+i) > max)
                max = *(a+offset+i);
        }
        if(active){
            if(c[i] < max) c[i] = max;
        }
    }
}

void cudaSend(int srcRank, int dstRank, int srcDevice, void* recvBuf, int recvOffset, 
                float* localBuf, int localOffset, int floatNum, cudaStream_t stream, 
                char* signal, int mpiTag, bool isCrossNode,
                cudaEvent_t event, bool* records, int idx, bool setRecord){
    int bytes = floatNum*4;
    if(!isCrossNode){
        // uva based p2p has lower performance
        // cudaMemcpyAsync(recvBuf+recvOffset, localBuf+localOffset, bytes,
        //         cudaMemcpyDeviceToDevice, stream);
        int dstDevice = dstRank - srcRank + srcDevice;
        cudaMemcpyPeerAsync(
            recvBuf+recvOffset, dstDevice, 
            localBuf+localOffset, srcDevice, 
            bytes, stream
        );        
        cudaEventRecord(event, stream);
        if(setRecord) records[idx] = true;
    }else{
        if(mpiTag < 0) ib_send(localBuf+localOffset);
        // is the last to set, but cross server
        if(setRecord) records[idx] = true; 
        MPI_Send(localBuf+localOffset, floatNum, MPI_FLOAT, dstRank,
                mpiTag, MPI_COMM_WORLD);
    }
}

void cudaRecv(int srcRank, float* recvBuf, int recvOffset, int floatNum,
              char* signal, int mpiTag, bool isCrossNode, cudaStream_t stream,
              cudaEvent_t event, bool* records, int idx){
    if(!isCrossNode){
        // intra node recv
        while(!records[idx]);
        cudaStreamWaitEvent(stream, event, 0);
    }else{
        float* recvBufOffset = recvBuf + recvOffset;
        if(mpiTag < 0) recvBufOffset = ib_recv();
        MPI_Recv(
            recvBufOffset, floatNum, MPI_FLOAT, 
            srcRank, mpiTag, MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE
        );
    }         
}

int createRecvSocket(int myRank, char* ip, int port){
    struct sockaddr_in stSockAddr;
    int fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    if(fd == -1){
        printf("[Rank%d]can not create recv socket.\n", myRank);
        exit(EXIT_FAILURE);
    }

    memset(&stSockAddr, 0, sizeof(struct sockaddr_in));

    stSockAddr.sin_family = AF_INET;
    stSockAddr.sin_port = htons(port);
    // stSockAddr.sin_addr.s_addr = INADDR_ANY;
    inet_pton(AF_INET, ip, &stSockAddr.sin_addr);

    if(bind(
        fd,(const struct sockaddr *)&stSockAddr, 
        sizeof(struct sockaddr_in)) == -1){
        printf("[Rank%d]can not bind socket fd.\n", myRank);
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    return fd;
}

int createSendSocket(int myRank, char* ip, int port){
    // connect(SocketFD, (const struct sockaddr *)&stSockAddr, sizeof(struct sockaddr_in))
    struct sockaddr_in stSockAddr;
    int fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    if(fd == -1){
        printf("[Rank%d]can not create send socket.\n", myRank);
        exit(EXIT_FAILURE);
    }

    memset(&stSockAddr, 0, sizeof(struct sockaddr_in));

    stSockAddr.sin_family = AF_INET;
    stSockAddr.sin_port = htons(port);
    int Res = inet_pton(AF_INET, ip, &stSockAddr.sin_addr);

    if (0 > Res){
        printf("[Rank%d]error: first parameter \
                    is not a valid address family\n", myRank);
        close(fd);
        exit(EXIT_FAILURE);
    }else if (0 == Res){
        printf("[Rank%d]error: char string second parameter \
                does not contain valid ipaddress\n", myRank);
        close(fd);
        exit(EXIT_FAILURE);
    }

    while(connect(fd, (const struct sockaddr *)&stSockAddr, 
            sizeof(struct sockaddr_in)) < 0);

    return fd;
}

int generatePort(int serverRank, int clientRank, char type){
    if(type == 'g')
        return (GLOBAL_SOCKET_START_PORT + 
                    100 * serverRank + clientRank);
    if(type == 'w')
        return (WORK_SOCKET_START_PORT + 
                    100 * serverRank + clientRank);
    else{
        printf("Wrong type in generatePort()\n");
        return -1;
    }
}

socketFds** buildIpcChannels(char** ipTable, int myRank, int nRanks){
    socketFds* gFds = (socketFds*)malloc(sizeof(socketFds_st));
    socketFds* wFds = (socketFds*)malloc(sizeof(socketFds_st));
    
    socketFds** res = (socketFds**)malloc(sizeof(socketFds*)*2);
    res[0] = gFds;
    res[1] = wFds;

    for(int i=0; i<nRanks; i++){
        int globalPort = generatePort(myRank, i, 'g');
        int workPort = generatePort(myRank, i, 'w');

        gFds->recvFd[i] = createRecvSocket(myRank, ipTable[myRank], globalPort);
        wFds->recvFd[i] = createRecvSocket(myRank, ipTable[myRank], workPort);
    }

    for(int i=0; i<nRanks; i++){
        // i as server
        for(int j=0; j<nRanks; j++){
            // build j --> i
            if(i == j) continue;
            if(i == myRank){
                listen(gFds->recvFd[j], 10);
                listen(wFds->recvFd[j], 10);
                // new fd used to read/recv
                gFds->recvFd[j] = accept(gFds->recvFd[j], NULL, NULL);
                wFds->recvFd[j] = accept(wFds->recvFd[j], NULL, NULL);
            }else if(j == myRank){
                int globalPort = generatePort(i, myRank, 'g');
                int workPort = generatePort(i, myRank, 'w');

                gFds->sendFd[i] = createSendSocket(myRank, ipTable[i], globalPort);
                wFds->sendFd[i] = createSendSocket(myRank, ipTable[i], workPort);

                // printf("[Rank%d]: create send socket on port %d\n", myRank, globalPort);
            }   
        }
    }

    printf("[Rank%d]Socket IPC Connections Built.\n", myRank);
    return res;
}

void socketIpcSend(int fd, char* buf, int size){
    send(fd, buf, size, 0);
}

void socketIpcRecv(int fd, char* buf, int size){
    recv(fd, buf, size, 0);
}

void attachSocketPort(int port){
    GLOBAL_SOCKET_START_PORT = port;
    WORK_SOCKET_START_PORT = port + 1000;
}
