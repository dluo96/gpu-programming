// Matrix multiplication kernel where each thread produces one output matrix column.
// Assumptions: square matrices. 

#include <stdio.h>
#include <cstdlib>
#include <cassert>

__global__ void matMulThreadCol(int *a, int *b, int *c, int N) {
    // Every thread computes one column in the output matrix.
    // Calculate global column of each thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (col < N) {
        // Each thread computes one column in the output matrix `c` and
        // thus traverses the rows (row=0, ..., N-1) of that column.
        for(int row = 0; row < N; row++) {
            // For the computation of each element, the thread 
            // traverses one row in `a` and one column in `b`. 
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = tmp;  // Write element to output matrix
        }
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

    // Allocate unified memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialise input matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Since each thread computes one column of the output `c`, we
    // only need `N` threads in total along the x-dimension
    int threadsPerBlockDim = 16;
    int blocksPerGridDim = (N + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 dimBlock(threadsPerBlockDim, 1);
    dim3 dimGrid(blocksPerGridDim, 1);

    // Invoke kernel
    matMulThreadCol<<<dimGrid, dimBlock>>>(a, b, c, N);

    // As we are not doing a Memcpy (a synchronizing operation) due to using `cudaMallocManaged`,
    // we need to synchronize another way:
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    printf("Success! Computed kernel where each thread produces one output matrix column.\n");
    return 0;
}
