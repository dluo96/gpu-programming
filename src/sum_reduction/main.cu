#include <iostream>
#include <cassert>
#include "sum_reduction_kernels.h"

void init_vector(int *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;  // Initializing with 1 for simplicity
    }
}

int main() {
    std::cout << "[1] Interleaved Addressing with Warp Divergence\n"
                 "[2] Interleaved Addressing with Shared Memory Bank Conflicts\n"
                 "[3] Sequential Addressing\n"
                 "[4] First Sum During Load from Global Memory\n"
                 "[5] Unrolling of the Last Warp using SIMD Execution\n"
                 "Enter the version of sum reduction kernel to use (1-5): ";
    int version;
    std::cin >> version;

    if (version < 1 || version > 5) {
        std::cerr << "Invalid version number. Please choose between 1 and 5." << std::endl;
        return EXIT_FAILURE;
    }

    // Size of array whose sum we will compute
    int N = 1 << 24;
    size_t bytes = N * sizeof(int);

    // Declare host arrays and device arrays
    int *input, *result;
    int *d_input, *d_result;

    // Allocate CPU and GPU memory
    input = (int*)malloc(bytes);
    result = (int*)malloc(bytes);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_result, bytes);

    // Populate array
    init_vector(input, N);

    // Copy to device
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    // CUDA events for timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Kernel decomposition
    int blockSize = SHMEM_LEN; // In number of threads
    int gridSize;
    cudaEventRecord(start);
    switch (version) {
        case 1:
            gridSize = (N/2 + blockSize - 1) / blockSize;
            sum_reduction_v1<<<gridSize, blockSize>>>(d_input, d_result, N); 
            break;
        case 2:
            gridSize = (N + blockSize - 1) / blockSize;
            sum_reduction_v2<<<gridSize, blockSize>>>(d_input, d_result, N); 
            break;
        case 3:
            gridSize = (N + blockSize - 1) / blockSize;
            sum_reduction_v3<<<gridSize, blockSize>>>(d_input, d_result, N);
            break;
        case 4:
            gridSize = (N/2 + blockSize - 1) / blockSize;
            sum_reduction_v4<<<gridSize, blockSize>>>(d_input, d_result, N);
            break;
        case 5:
            gridSize = (N/2 + blockSize - 1) / blockSize;
            sum_reduction_v5<<<gridSize, blockSize>>>(d_input, d_result, N);
            break;
    }

    // Track how many partial results are left to be added and perform kernel 
    // decomposition with recursion.
    // Note: although CUDA kernel launches are asynchronous, all GPU-related tasks
    // placed in one stream (the default behavior) are executed sequentially. 
    // Hence there is no need for `cudaDeviceSynchronize` between kernel calls here.
    unsigned int numRemain = gridSize;
    while(numRemain > 1) {
        switch (version) {
            case 1:
                gridSize = (numRemain + blockSize - 1) / blockSize;
                sum_reduction_v1<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                break;
            case 2:
                gridSize = (numRemain + blockSize - 1) / blockSize;
                sum_reduction_v2<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                break;
            case 3:
                gridSize = (numRemain + blockSize - 1) / blockSize;
                sum_reduction_v3<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                break;
            case 4:
                gridSize = (numRemain/2 + blockSize - 1) / blockSize;
                sum_reduction_v4<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                break;
            case 5:
                gridSize = (numRemain/2 + blockSize - 1) / blockSize;
                sum_reduction_v5<<<gridSize, blockSize>>>(d_result, d_result, numRemain);
                break;
        }
        numRemain = gridSize;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result to host
    cudaMemcpy(result, d_result, bytes, cudaMemcpyDeviceToHost);

    // Check result
    printf("Result: %d \n", result[0]);
    assert(result[0] == N);
    printf("Success! Computed sum reduction.\n");
    printf("Time elapsed: %f milliseconds.\n", milliseconds);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_result);
    free(input);
    free(result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
