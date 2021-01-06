default: conv

# CC = gcc-10
CC = g++

# FLAGS= -O3 -std=c++17 -Wall -g -fPIC -lm
FLAGS = -O3  -m64 -ffloat-store -w -g

OBJS = main.o kernel.o bmpfuncs.o  serialConv.o helper.o
NVCC = nvcc

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

conv: $(OBJS)
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) 


%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@
kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp