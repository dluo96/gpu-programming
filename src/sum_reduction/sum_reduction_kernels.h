#ifndef SUM_REDUCTION_KERNELS_H
#define SUM_REDUCTION_KERNELS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Shared memory on GPU
#define SHMEM_LEN 128
#define SHMEM_BYTES (SHMEM_LEN * 4)

__global__ void sum_reduction_v1(int *g_input, int *g_output, int numElements);
__global__ void sum_reduction_v3(int *g_input, int *g_output, int numElements);
__global__ void sum_reduction_v2(int *g_input, int *g_output, int numElements);
__global__ void sum_reduction_v4(int *g_input, int *g_output, int numElements);
__global__ void sum_reduction_v5(int *g_input, int *g_output, int numElements);
__device__ void warpReduce(volatile int* sdata, int ltid);

#endif
