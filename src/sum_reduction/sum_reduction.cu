// Different implementations of sum reduction
// with different optimization strategies

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "sum_reduction_kernels.h"

// #define SIZE 128
// #define SHMEM_SIZE 128 * 4

// Kernel v1: Interleaved Addressing with Divergent Branches.
// Disadvantages: thread divergence within warps are inefficient 
// and the modulo (%) operator is slow. 
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

// Kernel version 2: Interleaved Addressing with Bank Conflicts.
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
    // However, it introduces shared memory bank conflicts, which occur when
    // multiple threads in a given warp access different address locations
    // within the same bank. When this happens, the accesses serialize
    // rather than happening in parallel, thus reducing throughput.
    // Note: in NVIDIA GPUs, shared memory is divided into equally sized 
    // memory modules called banks. For many architectures, shared memory has
    // 32 banks, and each bank can service one memory request per clock cycle 
    // without conflicts.
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        // Index of element (in shared memory) to update
        unsigned int updateIdx = 2 * s * ltid;

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

// Kernel version 3: Sequential Addressing.
// Compared to Version 2, this kernel replaces the strided indexing
// in the inner loop with a reversed loop and thread-ID-based indexing.
// Advantages: the above means sequential addressing is conflict free.
// Disadvantages: half of the threads are idle on the 1st iteration, 
// three quarters of the threads are idle on the 2nd iteration, etc. 
__global__ void sum_reduction_v3(int *g_input, int *g_output, int numElements) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ltid = threadIdx.x;

    if (tid < numElements) {
        sdata[ltid] = g_input[tid];
    } else {
        sdata[ltid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory. 
    // Sequential addressing solves the shared memory bank conflicts
    // because the threads now access shared memory with a stride of
    // one 32-bit word (unsigned int) now. 
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(ltid < s) {
            sdata[ltid] += sdata[ltid + s];
        }
        __syncthreads();
    }

    // Write result for this block from shared to global memory
    if (ltid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }
}

// Kernel version 4: First Sum During Load from Global Memory
__global__ void sum_reduction_v4(int *g_input, int *g_output, int numElements) {
    __shared__ unsigned int sdata[SHMEM_SIZE];

    // Halve the number of thread blocks: instead of a single load,
    // each thread loads 2 elements from global memory, sums them, and
    // loads the result into shared memory.
    unsigned int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int ltid = threadIdx.x;
    if (tid < numElements) {
        // Do first sum during load
        sdata[ltid] = g_input[tid] + g_input[tid + blockDim.x];
    } else {
        sdata[ltid] = 0;
    }
    __syncthreads();

    // Same as version 3
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(ltid < s) {
            sdata[ltid] += sdata[ltid + s];
        }
        __syncthreads();
    }

    // Write result for this block from shared to global memory
    if (ltid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }
}

