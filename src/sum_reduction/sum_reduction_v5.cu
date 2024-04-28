// Version 5: unrolling last warp

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 128
#define SHMEM_SIZE 128 * 4

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
        __shared__ int sdata[SHMEM_SIZE];

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

void init_vector(int *v, int n) {
        for (int i = 0; i < n; i++) {
                v[i] = 1;
        }
}

int main() {
        int N = 1 << 24;
        size_t bytes = N * sizeof(int);

        int *input, *result;
        int *d_input, *d_result;

        // Allocate CPU and GPU memory, populate input, and copy to device
        input = (int*)malloc(bytes);
        result = (int*)malloc(bytes);
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_result, bytes);
        init_vector(input, N);
        cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

        // Block size (#threads) and grid size (#blocks)
        int blockSize = SIZE;
        int gridSize = (N/2 + blockSize - 1) / blockSize; // Division by 2 is for v4-v5

        // CUDA events for timing kernels
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;

        // Perform first kernel call
        cudaEventRecord(start);
        sum_reduction_v5<<<gridSize, blockSize>>>(d_input, d_result, N);

        // Track how many partial results are left to be added and perform kernel 
        // decomposition with recursion.
        // Note: although CUDA kernel launches are asynchronous, all GPU-related tasks
        // placed in one stream (the default behavior) are executed sequentially. 
        // Hence there is no need for `cudaDeviceSynchronize` between kernel calls here.
        unsigned int numRemain = gridSize;
        while(numRemain > 1) {
                gridSize = (numRemain/2 + blockSize - 1) / blockSize; // Division 2 is for v4-v5
                sum_reduction_v5<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                numRemain = gridSize;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy to host
        cudaMemcpy(result, d_result, bytes, cudaMemcpyDeviceToHost);

        // Check result
        printf("Result: %d \n", result[0]);
        assert(result[0] == N);
        printf("Success! Computed sum reduction.\n");
        printf("Time elapsed: %f milliseconds\n", milliseconds);

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_result);
        free(input);
        free(result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return 0;
}
