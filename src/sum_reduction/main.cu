#include <iostream>
#include <cassert>
#include "sum_reduction.h"

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

        // Allocate CPU and GPU memory, populate input, and copy to device
        input = (int*)malloc(bytes);
        result = (int*)malloc(bytes);
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_result, bytes);
        init_vector(input, N);
        cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

        // Block size (#threads) and grid size (#blocks)
        int blockSize = SHMEM_LEN;
        int gridSize = (N/2 + blockSize - 1) / blockSize; // Division by 2 is for v4-v5

        // CUDA events for timing kernels
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;

        // Perform first kernel call
        cudaEventRecord(start);
        sum_reduction_v4<<<gridSize, blockSize>>>(d_input, d_result, N);

        // Track how many partial results are left to be added and perform kernel 
        // decomposition with recursion.
        // Note: although CUDA kernel launches are asynchronous, all GPU-related tasks
        // placed in one stream (the default behavior) are executed sequentially. 
        // Hence there is no need for `cudaDeviceSynchronize` between kernel calls here.
        unsigned int numRemain = gridSize;
        while(numRemain > 1) {
                gridSize = (numRemain/2 + blockSize - 1) / blockSize; // Division 2 is for v4-v5
                sum_reduction_v4<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
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
