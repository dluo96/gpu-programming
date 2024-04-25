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

// Reduces a thread group to a single element.
// In particular, it performs an in-place reduction of values stored
// in shared memory.
__device__ int reduce_sum(cg::thread_group g, int *sdata, int val) {
    // Identify the thread within the group
    int lane = g.thread_rank();

    // The iteratively loop halves the number of threads active in summing
    // pairs of elements stored in shared memory, where each active thread 
    // adds a value from an idle thread located a stride (decreasing with
    // each iteration). This reduces the array to a single summed value.
    // Note: this is similar to the sequential addressing in version 3 of
    // the sum reduction kernel.
    for(int i = g.size() / 2; i > 0; i /= 2) {
        // Write to shared memory and ensure all threads have finished
        // before proceeding
        sdata[lane] = val;
        g.sync();

        // Conditionally add a value from a partner thread at a specific
        // stride and ensure all threads have finished before proceeding
        if (lane < i) {
            val += sdata[lane + i];
        }
        g.sync();
    }
    // Note: only thread 0 will return full sum
    return val; 
}

// Each thread computes a partial sum (4 elements at a time)
// from a subset of elements in the input array.
__device__ int thread_sum(int *input, int N) {
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // For the thread in question, loop over the elements in the
    // array that this thread is responsible for adding. Each 
    // iteration of the loop skips ahead by the total number of
    // threads in the grid. The n/4 comes from the fact that each
    // thread handles 4 elements at a time per loop iteration. 
    for (int i = tid; i < N / 4; i += blockDim.x * gridDim.x) {
        // With `int4`, we effectively read a block of four consecutive
        // integers from the array, starting at index 4*i. In particular, 
        //      input[4*i]     gives in.x
        //      input[4*i + 1] gives in.y
        //      input[4*i + 2] gives in.z
        //      input[4*i + 3] gives in.w
        // Using `int4` allows loading and adding four integers at a time,
        // improving memory throughput.
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_reduction(int *sum, int *input, int N) {
    // Each thread computes a partial sum
    int my_sum = thread_sum(input, N);

    // Dynamic shared memory allocation
    extern __shared__ int sdata[];

    // In this case, a thread group is a thread block
    auto g = cg::this_thread_block();

    // Each block adds the partial sum computed by each of 
    // its threads. This returns another partial sum.
    int block_sum = reduce_sum(g, sdata, my_sum);

    // Add the partial sum from each block.
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

    // Allocation in unified memory
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&input, bytes);

    // Populate the array
    init_vector(input, N);

    // Sizes of blocks, grid, and dynamic shared memory
    // With the grid size specified, each thread would
    // process exactly one `int4` (see `thread_sum` above).
    // To handle multiple, we could decrease the grid size.
    int blockSize = 128;
    int quadrupletsPerThread = 2; // Number of `int4` processed per thread
    int gridSize = (N/(4 * quadrupletsPerThread) + blockSize - 1) / blockSize;
    size_t sharedBytes = blockSize * sizeof(int);

    // CUDA events for timing kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Call kernel with dynamic shared memory
    cudaEventRecord(start);
    sum_reduction<<<gridSize, blockSize, sharedBytes>>>(sum, input, N);
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
