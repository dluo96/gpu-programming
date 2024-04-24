// Cooperative Groups extends the CUDA programming model 
// to provide flexible, dynamic grouping of threads.
// 
// With GPU-wide sync (and not just block-wide sync with
// __syncthreads) we no longer need kernel decomposition. 

// Kernel version 6: sum reduction using Cooperative Groups.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

namespace cg = cooperative_groups;

// Reduces a thread group to a single element
__device__ int reduce_sum(cg::thread_group g, int *temp, int val){
    int lane = g.thread_rank();

    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2){
        temp[lane] = val;
        // Wait for all threads to store
        g.sync();
        if (lane < i) { val += temp[lane + i]; }
        // Wait for all threads to load
        g.sync();
    }
    // Note: only thread 0 will return full sum
    return val; 
}

// Creates partials sums from the original array
__device__ int thread_sum(int *input, int n){
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // For the thread in question, loop over the elements in the
    // array that this thread is responsible for adding. Each 
    // iteration of the loop skips ahead by the total number of
    // threads in the grid. The n/4 comes from the fact that each
    // thread handles 4 elements at a time per loop iteration. 
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) {
        // With `int4`, we effectively read a block of four consecutive
        // integers from the array, starting at index 4*i. In particular, 
        //      in.x input[4*i] in.x
        //      input[4*i + 1] as in.y
        //      input[4*i + 2] as in.z
        //      input[4*i + 3] as in.w
        // Using `int4` allows loading and adding four integers at a time,
        // improving memory throughput.
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_reduction(int *sum, int *input, int n){
    // Create partial sums from the array
    int my_sum = thread_sum(input, n);

    // Dynamic shared memory allocation
    extern __shared__ int temp[];

    // Identifier for a block
    auto g = cg::this_thread_block();

    // Reudce each block
    int block_sum = reduce_sum(g, temp, my_sum);

    // Collect the partial result from each block.
    // Here `atomicAdd()` reads a word at an address 
    // in shared memory, adds a number to it, and
    // writes the result back to the same address.
    if (g.thread_rank() == 0) {
            atomicAdd(sum, block_sum);
    }
}

void init_vector(int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1; //rand() % 10;
    }
}

int main() {
    int N = 1 << 24;
    size_t bytes = N * sizeof(int);
    int *sum, *input;

    // Allocate using unified memory
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&input, bytes);

    init_vector(input, N);

    // Sizes of blocks, grid, and dynamic shared memory
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t shmemSize = blockSize * sizeof(int);

    // CUDA events for timing kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Call kernel with dynamic shared memory
    cudaEventRecord(start);
    sum_reduction<<<gridSize, blockSize, shmemSize>>>(sum, input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    printf("Result is %d \n", sum[0]);
    assert(*sum == N);
    printf("Success! Computed sum reduction.\n");
    printf("Time elapsed: %f milliseconds\n", milliseconds);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
