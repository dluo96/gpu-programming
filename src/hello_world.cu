#include <iostream>

void helloCPU()
{
  std::cout<<"Hello, World! (CPU)\n";
}

__global__
void helloGPU()
{
  printf("Hello, World! (GPU - Block %i | Thread %i)\n", blockIdx.x, threadIdx.x);
}

int main()
{
  helloCPU();

  // Set number of blocks and number of threads per block
  int numBlocks = 2;
  int threadsPerBlock = 4;

  // Invoke kernel
  helloGPU<<<numBlocks, threadsPerBlock>>>();

  // Force the host (CPU) to wait for the GPU - this ensures that all GPU 
  // operations are completed before the program exits, which is especially
  // important here to make sure we see all the printed messages from the 
  // GPU before the program terminates
  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}