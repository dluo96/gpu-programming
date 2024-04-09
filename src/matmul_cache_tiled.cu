// This program computes matrix multiplication (of square matrices)
// on the GPU using cache tiling and shared memory (user managed L1 cache)
// to access global memory less often. 
// See the README for further explanations of what shared memory is. 

#include <cstdlib>
#include <cassert>
#include <iostream>

#define SHMEM_SIZE (16 * 16)

// Compared to before, we will do in 16-element (size of our thread blocks) chunks. 
// Previously we read from main memory, where we hoped to get lucky in cache hierarchy.
// Here we guarantee fast memory by using shared memory. 

__global__ void matMulCacheTiled(int *a, int *b, int *c, int N) {
    // Shared memory is something we have to allocate (like global memory)
    // We can do it dynamically: at runtime, we decide how miuch shared memory to allocate, or
    // Static allocation: i.e. at compile time: compiler needs to know how much shared memory we will use. 
    // The latter can be easier to work, but it's not always possible because we might not need how much memory we need until at runtime. 
    //  In this example, we know it a priori.

    // Allocate two statically-sized pieces of shared memory
    // NB: shared memory is private per thread block
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile = size of a thread block, 16x16
    // We load this into a small matrix

    // Define variables for convenience
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x; // Could be x or y, they are equal

    // Move the tile across the length of the grid.
    // The number of tiles along one dimension (x or y) is N/dim. 
    // Padding it leads to (N + dim - 1) / dim. 
    // Loop over tiles:
    int tmp = 0;
    for (int i = 0; i < ((N + dim - 1) / dim); i++) {
        // Thread index within given block
        int threadIdxInBlock = ty * dim + tx;

        // Across rows
        // row * N tells us which row we are at
        // i * dim tells us which tile we are at
        // tx tells us the thread index (within the block)
        A[threadIdxInBlock] = a[(row * N) + (i * dim) + tx];

        // Across columns
        // i * dim * N traverses complete row(s) of blocks
        // ty * N traverses complete rows within a single row of blocks
        // col traverses a row
        B[threadIdxInBlock] = b[(i * dim * N) + (ty * N) + col];

        // NB: in the above we still have to access global memory. 
        // But we only have to do it once per thread block. 

        // Before calculating values, ensure that every thread within 
        // the thread block has loaded into shared memory.
        // This is crucial threads will use the values loaded by other threads
        __syncthreads();

        // Calculate all temporary values for this tile
        // Recall we are splitting our NxN matrix multiplication into 
        // chunks of 16x16 multiplication
        // Accumulate the partial results
        for (int j = 0; j < dim; j++) {
            tmp += A[ty * dim + j] * B[j * dim + tx];
        }
        // Ensure all threads are done with values in shared memory
        // before loading in new ones
        __syncthreads();

        // Write result back to main memory
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

    printf("Program completed successfully!\n");
    return 0;
}