// This programs implement a CUDA C/C++ kernel for
// a 1D convolution utilising constant memory for
// increased performance. 

#include <stdio.h>
#include <cstdlib>
#include <cassert>

#define MASK_LEN 7

// Allocate constant memory: `MASK` will be visible inside the kernel.
// Using constant memory means we avoid cache misses.
__constant__ int MASK[MASK_LEN];

// Do not need to pass mask to function since `MASK` is available
__global__ void convolution_1d_constant_memory(int *input, int *output, int N) {
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if(tid < N) {
        int radius = MASK_LEN / 2;
        int start = tid - radius;
        int tmp = 0;
        // Iterate over length of the mask
        for(int i = 0; i < MASK_LEN; i++) {
            int arrayIdx = start + i;
            // Check if we are off either edge of the input array
            if((arrayIdx >= 0) && (arrayIdx < N)) {
                tmp += input[arrayIdx] * MASK[i];
            }
        }
        output[tid] = tmp;
    }
}

// Verify the result on the CPU
void verify_result(int *input, int *mask, int *output, int N, int M) {
    int radius = M / 2;
    int tmp;
    int start;
    for(int i = 0; i < N; i++) {
        start = i - radius;
        tmp = 0;
        for(int j = 0; j < M; j++) {
            int arrayIdx = start + j;
            // Ignore out-of-bound elements of input array
            if((arrayIdx >= 0) && (arrayIdx < N)) {
                tmp += input[arrayIdx] * mask[j];
            }
        }
        assert(tmp == output[i]);
    }
}

// Initialise array with random numbers
void init_array(int *array, int N) {
    for(int i = 0; i < N; i++) {
        array[i] = rand() % 100;
    }
}

int main() {
    // Size of input (and output) array
    int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    // Allocate space on host for input and output arrays
    int *input, *output;
    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&output, bytes);

    // Initialise input array
    init_array(input, N);

    // Allocate space on host for mask
    int *mask = new int[MASK_LEN];
    init_array(mask, MASK_LEN);

    // Because we have allocated space for the mask 
    // on the GPU directly, we have to directly copy it.
    // This is unlike the unified memory for `cudaMallocManaged`
    // where the memory is paged. 
    cudaMemcpyToSymbol(MASK, mask, MASK_LEN * sizeof(int));

    // Dimensions for thread blocks and grid
    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Launch kernel and synchronize
    convolution_1d_constant_memory<<<BLOCKS, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();

    // Verify result
    verify_result(input, mask, output, N, MASK_LEN);

    printf("Success! Computed 1D convolution optimised with constant memory.\n");

    return 0;
}