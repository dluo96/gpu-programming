// Version 5: unrolling last warp

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

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

__global__ void sum_reduction(int *g_input, int *g_output, int len) {
        // Allocate shared memory
        __shared__ int sdata[SHMEM_SIZE];

        int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        int ltid = threadIdx.x;

        // Perform first addition during load from global to shared memory
        if(tid < len) {
                if(gridDim.x > 1) {
                        sdata[ltid] = g_input[tid] + g_input[tid + blockDim.x];
                } else { // If there is only one thread block
                        sdata[ltid] = g_input[tid]; 
                }
        } else {
                sdata[ltid] = 0;
        }
        __syncthreads();

        // Start at 1/2 block stride and divide by two each iteration
        // Stop early (call device function instead)
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
                // Each thread does work unless it is further than the stride
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

void initialize_vector(int *v, int n) {
        for (int i = 0; i < n; i++) {
                v[i] = 1;
        }
}

int main() {
        int n = 1 << 28;
        size_t bytes = n * sizeof(int);

        int *h_v, *h_v_r;
        int *d_v, *d_v_r;

        // Allocate memory
        h_v = (int*)malloc(bytes);
        h_v_r = (int*)malloc(bytes);
        cudaMalloc(&d_v, bytes);
        cudaMalloc(&d_v_r, bytes);

        // Initialize vector
        initialize_vector(h_v, n);

        // Copy to device
        cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

        // TB Size
        int TB_SIZE = SIZE;

        // First time
        int GRID_SIZE = (n/2 + TB_SIZE - 1) / TB_SIZE;

        printf("Elements remaining: %i.\n", n);
        sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r, n);
        printf("Kernel launched with %i blocks\n", GRID_SIZE);

        int numRemain = GRID_SIZE;
        while(numRemain > 1) {
                int GRID_SIZE = (numRemain/2 + TB_SIZE - 1) / TB_SIZE;
                printf("Elements remaining: %i.\n", numRemain);
                sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v_r, d_v_r, numRemain);
                printf("Blocks launched: %i\n", GRID_SIZE);
                cudaDeviceSynchronize();
                numRemain = GRID_SIZE;
        }

        // Copy to host;
        cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

        // Print the result
        printf("Result: %d \n", h_v_r[0]);
        assert(h_v_r[0] == n);
        printf("Success! Completed sum reduction.\n");

        return 0;
}