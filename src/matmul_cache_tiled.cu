// CUDA C/C++ kernel for computing matrix multiplication (of square matrices)
// on the GPU using cache tiling and shared memory (user managed L1 cache).
// See `README.md` for further explanations of shared memory and how it compares
// to global memory. 

#include <cstdlib>
#include <cassert>
#include <iostream>

// Compared to the naive implementation, we will perform the computation in 16x16 
// (size of our thread blocks) chunks. Previously we read from global memory, 
// hoping to get lucky in cache hierarchy. Here we guarantee fast memory access
// by using shared memory. 
#define SHMEM_SIZE (16 * 16)

__global__ void matMulCacheTiled(int *a, int *b, int *c, int N) {
    // Allocate two statically-sized pieces of shared memory
    // NB: shared memory is private per thread block
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Convenience variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x; // Could be x or y, they are equal by assumption
 
    // Outer loop over tiles
    // Number of tiles along one dimension (x or y) is N/dim.
    // Padding this gives (N + dim - 1) / dim. 
    int tmp = 0;
    for (int i = 0; i < ((N + dim - 1) / dim); i++) {
        // Thread index within the given block
        int threadIdxInBlock = ty * dim + tx;

        /* Explanation of loading from global memory into shared memory.

        Each thread (in the block) loads:
            One element from `a` (global memory) into `A` (shared memory).
            One element from `b` (global memory) into `B` (shared memory).

        For `A`:
            row * N tells us which row we are at,
            i * dim tells us which tile we are at,
            tx tells us the thread index (within the block).
        
        For `B`:
            i * dim * N traverses complete row(s) of blocks,
            ty * N traverses complete rows within a single row of blocks,
            col traverses a row.
        
        Note that while this still requires accessing global memory, it only
        has to be done once for each element of `a` and `b`. This is because
        the threads in a thread block can access the shared memory private to
        that block.
        */
        A[threadIdxInBlock] = a[(row * N) + (i * dim) + tx];
        B[threadIdxInBlock] = b[(i * dim * N) + (ty * N) + col];

        // Before calculating values, ensure that every thread in the block has
        // loaded its values (from `a` and `b`) into shared memory (`A` and `B`).
        // This is crucial because every thread will use values loaded by other 
        // threads in its computation.
        __syncthreads();

        // Calculate partial result (dot product of row in `A` and column in `B`
        // and accumulate.
        for (int j = 0; j < dim; j++) {
            tmp += A[ty * dim + j] * B[j * dim + tx];
        }

        // Ensure all threads have finished their calculations and thus no longer
        // need the current values in shared memory.
        __syncthreads();

        // Write result back to main (global) memory
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
    // Set the matrix dimensions
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialise our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set sizes of thread blocks and grid
    int threadsPerBlockDim = 16;
    int blocksPerGridDim = (N + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 THREADS(threadsPerBlockDim, threadsPerBlockDim);
    dim3 BLOCKS(blocksPerGridDim, blocksPerGridDim);

    // Launch kernel
    matMulCacheTiled<<<BLOCKS, THREADS>>>(a, b, c, N);

    // Need to synchronize since kernel launch is async
    cudaDeviceSynchronize();

    // Verify the result on the CPU
    verify_result(a, b, c, N);

    printf("Success! Computed matrix multiplication optimised with cache tiling and shared memory.\n");
    return 0;
}