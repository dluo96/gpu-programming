// Version 5: unrolling last warp

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "sum_reduction.h"

// This function is a callable from the GPU - it can be thought of as helper
// function for the kernel. `volatile` is specified to prevent caching in 
// registers (a compiler optimization).
// Threads within a single warp execute in lockstep: all threads within the
// warp execute the same instruction at the same time. Due to this synchronous
// execution, this function, which involves shared memory where each thread 
// reads and writes to its own distinct memory location, does not require
// explicit synchronization mechanisms such as `__syncthreads()`. 
__device__ void warpReduce(volatile int* sdata, int ltid) {
        sdata[ltid] += sdata[ltid + 32]; // Executed in lockstep by all threads in the warp
        sdata[ltid] += sdata[ltid + 16]; // Executed in lockstep by all threads in the warp
        sdata[ltid] += sdata[ltid + 8]; // Executed in lockstep by all threads in the warp
        sdata[ltid] += sdata[ltid + 4]; // Executed in lockstep by all threads in the warp
        sdata[ltid] += sdata[ltid + 2]; // Executed in lockstep by all threads in the warp
        sdata[ltid] += sdata[ltid + 1]; // Executed in lockstep by all threads in the warp
}

__global__ void sum_reduction_v5(int *g_input, int *g_output, int numElements) {
        // Allocate shared memory
        __shared__ int sdata[SHMEM_BYTES];

        unsigned int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        unsigned int ltid = threadIdx.x;

        // Perform first addition during load from global to shared memory
        if(tid < numElements) {
                if(tid + blockDim.x < numElements) {
                        sdata[ltid] = g_input[tid] + g_input[tid + blockDim.x];
                } else {
                        sdata[ltid] = g_input[tid]; 
                }
        } else {
                sdata[ltid] = 0;
        }
        __syncthreads();

        // Stop early since `warpReduce` will take care of the rest
        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (ltid < s) {
                        sdata[ltid] += sdata[ltid + s];
                }
                __syncthreads();
        }

        if (ltid < 32) {
                warpReduce(sdata, ltid);
        }

        // Let thread 0 within each block write the result of the block
        // from global to shared memory
        if (ltid == 0) {
                g_output[blockIdx.x] = sdata[0];
        }
}
