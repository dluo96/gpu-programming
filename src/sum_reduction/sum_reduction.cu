// Sum Reduction: 7 versions (implementations) 
// with different optimization strategies

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

// Kernel v1: Interleaved Addressing with Divergent Branches.
// Drawbacks: highly divergent warps are inefficient and the 
// modulo (%) operator is slow. 
__global__ void sum_reduction_v1(int *g_input, int *g_output, int numElements) {
    // Allocate dynamic shared memory
    extern __shared__ unsigned int sdata[];

    // Global (relative to grid) and local (relative to block) thread IDs
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ltid = threadIdx.x;

    // Each thread loads one element from global to shared memory
     if (tid < numElements) {
        sdata[ltid] = g_input[tid];
    } else {
        sdata[ltid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory. Stride is doubled every iteration
    // (halving the number of threads used within the given block).
    // The interleaved addressing leads to large thread divergence because
    // threads are active/idle depending on whether their thread IDs are
    // powers of 2. The if-statement causes thread divergence within a warp. 
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if(ltid % (2 * s) == 0) {
            sdata[ltid] += sdata[ltid + s];
        }
        __syncthreads();
    }

    // Write result for this block from shared to global memory
    if (ltid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }
}

// Version 2: Interleaved Addressing with Bank Conflicts.
// Compared to Version 1, this kernel replaces the divergent
// branch in the inner loop with a strided index and non-divergent 
// branch. This leads to a new drawback: shared memory bank conflicts. 
__global__ void sum_reduction_v2(int *g_input, int *g_output, int numElements) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ltid = threadIdx.x;

     if (tid < numElements) {
        sdata[ltid] = g_input[tid];
    } else {
        sdata[ltid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory. Still uses interleaved addressing,
    // but threads being active/idle no longer depends on whether thread IDs
    // are powers of 2. Consecutive thread IDs now run, solving the issue of
    // threads diverging within a warp. 
    // However, it introduces bank conflicts in shared memory: ...
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int updateIdx = 2 * s * ltid; // Index of element to update

        if(updateIdx < blockDim.x) {
            sdata[updateIdx] += sdata[updateIdx + s];
        }
        __syncthreads();
    }

    // Write result for this block from shared to global memory
    if (ltid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }
}

// Version 3: Sequential Addressing.
// Compared to Version 2, this kernel replaces the
// strided indexing in the inner loop with a reversed
// loop and thread-ID-based indexing.
// Advantages: sequential addressing is conflict free.
// Disadvantages: half of the threads are idle on the 
// first loop of the iteration. 


void init_vector(int *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;  // Initializing with 1 for simplicity
    }
}

int main() {
    int N = 1 << 24;
    size_t bytes = N * sizeof(int);

    int *input, *result;
    int *d_input, *d_result;
    input = (int*)malloc(bytes);
    result = (int*)malloc(bytes);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_result, bytes);
    init_vector(input, N);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    // Threads, blocks, grids, and dynamic shared memory
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sizeSharedMemory = threadsPerBlock * sizeof(int);

    // CUDA events for timing kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Kernel decomposition with recursion
    cudaEventRecord(start);
    sum_reduction_v2<<<blocksPerGrid, threadsPerBlock, sizeSharedMemory>>>(d_input, d_result, N);
    cudaDeviceSynchronize();
    unsigned int numPartialSums = blocksPerGrid;
    while(numPartialSums > 1) {
        int nBlocks = (numPartialSums + threadsPerBlock - 1) / threadsPerBlock;
        // printf("Partial sums computed = %i. Threads per block = %i. Blocks required = %i.\n", 
        //         numPartialSums, threadsPerBlock, nBlocks);
        sum_reduction_v2<<<nBlocks, threadsPerBlock, sizeSharedMemory>>>(d_result, d_result, numPartialSums);
        cudaDeviceSynchronize();
        numPartialSums = nBlocks;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result of sum reduction v1: %d\n", result[0]);
    printf("Elapsed time: %f milliseconds\n", milliseconds);
    assert(result[0] == N);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_result);
    free(input);
    free(result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
