// CUDA C/C++ implementation of 1D convolution.

#include <stdio.h>
#include <cstdlib>
#include <cassert>

// 1D convolution kernel where each thread computes
// one elements of the output array.
//
// Arguments:
//      array = padded input array
//      mask = convolution mask
//      result = output array
//      N = number of elements in `array` and `result`
//      M = number of elements in `mask`
__global__ void convolution_1d(int *array, int *mask, int *result, int N, int M) {
    // Global thread ID
    tid = blockIdx.x * blockDim.x + threadIdx.x

    // Radius of convolution mask
    int radius = M / 2;

    // Index of first element (of the input array)
    // that is needed by the thread in question
    start = tid - radius;

    // Iterate over elements of mask
    int tmp = 0;
    for(int j = 0; j < M; j++) {
        arrayIdx = start + j; 
        // Ignore out-of-bound elements of input array
        if((arrayIdx >= 0) && (arrayIdx < N)) {
            tmp += array[arrayIdx] * mask[j];
        }
    }

    // Write result to output array
    result[tid] = tmp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int N, int M) {
    int radius = M / 2;
    int tmp;
    int start;
    for(int i = 0; i < N; i++) {
        start = i - radius;
        tmp = 0;
        for(int j = 0; j < M; j++) {
            arrayIdx = start + j;
            // Ignore out-of-bound elements of input array
            if((arrayIdx >= 0) && (arrayIdx < N)) {
                tmp += array[arrayIdx] * mask[j];
            }
        }
        assert(result[i] == tmp);
    }
}

int main() {
    // Size of input (and output) array
    int N = 1 << 10;
    size_t bytes = N * sizeof(int);

    // Size of convolution mask
    int M = 7;
    size_t maskBytes = M * sizeof(int);

    // Allocate host memory
    int *array = new int[N];
    int *mask = new int[M];
    int *result = new int[N];

    // Initialise
    for(int i = 0; i < n; i++) {
        array = rand() % 100;
    }
    for(int i = 0; i < M; i++) {
        mask[i] = rand() % 10;
    }    

    // Allocate device memory
    cudaMallocManaged(&array, bytes);
    cudaMallocManaged(&mask, maskBytes);
    cudaMallocManaged(&result, bytes);

    // Define threads per block and number of blocks
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Invoke kernel
    convolution_1d<<<blocks, threads>>>(array, mask, result, N, M);

    // Since using `cudaMallocManaged`, we call a sync operation
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(array, mask, result, N, M);

    printf("Successfully computed 1D convolution!\n");
    return 0;
}


