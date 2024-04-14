// CUDA C/C++ implementation of GPU kernel for matrix multiplication of square matrices.


#include <stdio.h>
#include <cstdlib>
#include <cassert>

__global__ void matMul(int *a, int *b, int *c, int N) {
    // Every thread computes one element in the output matrix.
    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for our matrix
    if (row < N && col < N) {

        // As every thread computes one element in the output matrix `c`,
        // every thread traverses one row in `a` and one column in `b`. 
        int tmp = 0;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k * N + col];
        }

        // Write element to output matrix
        c[row * N + col] = tmp;
    }
}

// Initialise a square matrix with random numbers between 0-100
void init_matrix(int *m, int N)
{
    for (int i = 0; i < N * N; i++) {
        m[i] = rand() % 100;
    }
}

// Verify the result of the GPU kernel with a CPU calculation
void verify_result(int *a, int *b, int *c, int N)
{
    int tmp;
    // For every row
    for(int i = 0; i < N; i++) {
        // For every column
        for(int j = 0; j < N; j++) {
            // For every element in the row-col pair ij
            tmp = 0;
            for(int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check each element
            assert(tmp == c[i * N + j]);
        }
    }
}

int main() {
    // Set square matrix dimensions (2^10 x 2^10 here)
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate host memory
    int *a, *b, *c;

    // Allocate device memory
    // With `cudaMallocManaged` (as opposed to `cudaMalloc`), the CUDA runtime 
    // manages the transfer of memory back and forth for you, so memcpy isn't needed
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialise our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Make a 2D grid (16x16) of blocks, where each block is itself a 2D grid of threads (16x16).
    // This makes the indexing of the output matrix more intuitive. We use the `dim3` CUDA type to do this.
    // Note that we need to calculate the number of blocks needed in each dimension to cover the entire 
    // output matrix of size N x N.
    int threadsPerBlockDim = 16;
    int blocksPerGridDim = (N + threadsPerBlockDim - 1);
    dim3 THREADS(threadsPerBlockDim, threadsPerBlockDim);
    dim3 BLOCKS(blocksPerGridDim, blocksPerGridDim);

    // Invoke matrix multiplication kernel: every launched thread
    // will calculate one element of the resulting matrix
    matMul<<<BLOCKS, THREADS>>>(a, b, c, N);

    // As we are not doing a Memcpy (a synchronizing operation) due to using `cudaMallocManaged`,
    // we can instead:
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    printf("Success! Computed matrix multiplication (naive implementation).\n");
    return 0;
}


