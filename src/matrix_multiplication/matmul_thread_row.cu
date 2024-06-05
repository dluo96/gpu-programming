// Matrix multiplication kernel where each thread produces one output matrix row.
// Assumptions: inputs are square matrices. 

#include <stdio.h>
#include <cstdlib>
#include <cassert>

__global__ void matMulThreadRow(int *a, int *b, int *c, int N) {
    // Every thread computes one row in the output matrix.
    // Calculate global row of each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (row < N) {
        // Each thread computes one row in the output matrix `c` and
        // thus traverses the columns (col=0, ..., N-1) of that row.
        for(int col = 0; col < N; col++) {
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

    // Since each thread computes one row of the output `c`, we
    // only need `N` threads in total along the y-dimension
    int threadsPerBlockDim = 16;
    int blocksPerGridDim = (N + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 dimBlock(1, threadsPerBlockDim);
    dim3 dimGrid(1, blocksPerGridDim);

    // Invoke kernel
    matMulThreadRow<<<dimGrid, dimBlock>>>(a, b, c, N);

    // As we are not doing a Memcpy (a synchronizing operation) due to using `cudaMallocManaged`,
    // we need to synchronize another way:
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    printf("Success! Computed kernel where each thread produces one output matrix row.\n");
    return 0;
}
