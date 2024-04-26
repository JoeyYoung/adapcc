.PHONY:clean

CC = nvcc
OMPI_LIB_DIR ?= /home/xyzhao/openmpi/lib
OMPI_INC_DIR ?= /home/xyzhao/openmpi/include
PRJ_INC_DIR ?= ./csrc/include

CFLAGS = -L$(OMPI_LIB_DIR) -I$(OMPI_INC_DIR) -I$(PRJ_INC_DIR) \
			-libverbs -lmpi -lpthread -lnuma -Xcompiler \
			-fPIC -shared

SRC_FILES := $(wildcard ./csrc/*.c) \
			$(wildcard ./csrc/*.cu) \
			$(wildcard ./csrc/*.cpp)

TARGET_FILE = communicator.so

${TARGET_FILE}: ${SRC_FILES}
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f ${TARGET_FILE}