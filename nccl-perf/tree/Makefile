.PHONY:clean

CC = nvcc
NCCL_LIB_DIR ?= /home/xyzhao/nccl/build/lib
NCCL_INC_DIR ?= /home/xyzhao/nccl/build/include
OMPI_LIB_DIR ?= /home/xyzhao/openmpi/lib
OMPI_INC_DIR ?= /home/xyzhao/openmpi/include

BUILDDIR ?= ./build

CFLAGS = -L$(NCCL_LIB_DIR) -L$(OMPI_LIB_DIR) -I$(NCCL_INC_DIR) -I$(OMPI_INC_DIR) -lnccl -lmpi

# SRC_FILES := $(wildcard *.cu)
# OBJ_FILES := $(SRC_FILES:%.cu=%.o)
# TARGET_FILES := $(SRC_FILES:%.cu=%)

# all:$(TARGET_FILES)

# $(TARGET_FILES):$(SRC_FILES)
# 	$(CC) $(CFLAGS) $< -o $@

# clean:
# 	rm -f $(OBJ_FILES) $(TARGET_FILES)

SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${BUILDDIR}/%.o)
BIN_FILES := $(SRC_FILES:%.cu=${BUILDDIR}/%)

all: ${BIN_FILES}

${BUILDDIR}/%: %.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f ${BUILDDIR}/*

# ${BUILDDIR}/%.o: %.cu
# 	$(CC) -o $@ $(CFLAGS) -c $<

# ${BUILDDIR}/%: ${BUILDDIR}/%.o
# 	$(CC) -o $@ $(CFLAGS) $<


