.PHONY:clean

CC = nvcc
OMPI_LIB_DIR ?= /home/xyzhao/openmpi/lib
OMPI_INC_DIR ?= /home/xyzhao/openmpi/include
PRJ_INC_DIR ?= ./include

BUILDDIR ?= .

CFLAGS = -L$(OMPI_LIB_DIR) -I$(OMPI_INC_DIR) -I$(PRJ_INC_DIR) -libverbs -lmpi -lpthread 

SRC_FILES := flow_dedicate.cu
BIN_FILES := ${BUILDDIR}/flow_dedicate

all: ${BIN_FILES}

${BIN_FILES}: ${SRC_FILES}
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f ${BIN_FILES}