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

using namespace cooperative_groups;

// Reduces a thread group to a single element
__device__ int reduce_sum(thread_group g, int *temp, int val){
    int lane = g.thread_rank();

    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2){
        temp[lane] = val;
        // Wait for all threads to store
        g.sync();
        if (lane < i) {
                val += temp[lane + i];
        }
        // Wait for all threads to load
        g.sync();
    }
    // note: only thread 0 will return full sum
    return val; 
}

// Creates partials sums from the original array
__device__ int thread_sum(int *input, int n){
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x){
        // Using `int4` allows loading and adding four integers
        // at a time, improving memory throughput
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

    // Identifier for a TB
    auto g = this_thread_block();

    // Reudce each TB
    int block_sum = reduce_sum(g, temp, my_sum);

    // Collect the partial result from each TB
    if (g.thread_rank() == 0) {
            atomicAdd(sum, block_sum);
    }
}

void initialize_vector(int *v, int n) {
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

    initialize_vector(input, N);

    // Blocks, grid, and size of dynamic shared memory
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t shmemSize = blockSize * sizeof(int);

    // Call kernel with dynamic shared memory
    sum_reduction<<<gridSize, blockSize, shmemSize>>>(sum, input, N);
    cudaDeviceSynchronize();

    printf("Result is %d \n", sum[0]);
    assert(*sum == N);
    printf("Success! Computed sum reduction.\n");

    return 0;
}
