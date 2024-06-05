// Matrix-vector multiplication kernel where each thread calculates an output vector element.
// Assumptions: input matrix is square.
// Note: Exercise 2. in Chapter 3 of the PMPP book.

#include <stdio.h>
#include <cstdlib>
#include <cassert>

__global__ void matMul(int *a, int *b, int *c, int N) {
    // Every thread computes one element in the output vector.
    // Calculate global row for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check for our matrix
    if (row < N) {
        // Because every thread computes one element in the output vector `c`,
        // every thread traverses one row in `a` and the (single) column in `b`. 
        int tmp = 0;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k];
        }

        // Write element to output matrix
        c[row] = tmp;
    }
}

// Initialise a square matrix with random numbers between 0-100
void init_matrix(int *m, int N)
{
    for (int i = 0; i < N * N; i++) {
        m[i] = rand() % 100;
    }
}

// Initialise a vector with random numbers between 0-100
void init_vector(int *m, int N)
{
    for (int i = 0; i < N; i++) {
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
                tmp += a[i * N + k] * b[k];
            }

            // Check each element
            assert(tmp == c[i]);
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

    // Initialise input matrix and input vector
    init_matrix(a, N);
    init_vector(b, N);

    // Each thread computes one element of the output vector `c`
    // Thus, a total of `N` threads are needed
    int threadsPerBlockDim = 16;
    int blocksPerGridDim = (N + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 dimBlock(1, threadsPerBlockDim);
    dim3 dimGrid(1, blocksPerGridDim);

    // Invoke kernel
    matMul<<<dimGrid, dimBlock>>>(a, b, c, N);

    // As we are not doing a Memcpy (a synchronizing operation) due to using `cudaMallocManaged`,
    // we need to synchronize another way:
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    printf("Success! Computed matrix vector multiplication.\n");
    return 0;
}


