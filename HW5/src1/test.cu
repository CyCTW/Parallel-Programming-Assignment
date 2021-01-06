#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#define INPUT_X -0.0612500f
#define INPUT_Y -0.9916667f

int diverge_cpu(float c_re, float c_im, int max)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__device__ int diverge_gpu(float c_re, float c_im, int max)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void kernel(int *c, int n)
{
  // 取得 global ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // 通通設一樣的值
  c[id] = diverge_gpu(INPUT_X, INPUT_Y, 256);
}

int main(int argc, char *argv[])
{
  int n = 1024*1024;
  int *h_c;
  int *d_c;
  h_c = (int *)malloc(n * sizeof(int));
  cudaMalloc(&d_c, n * sizeof(int));

  int blockSize = 1024;
  int gridSize = 1;

  // 這邊是算 GPU 的部分
  kernel<<<gridSize, blockSize>>>(d_c, n);
  cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

  // 這邊是算 CPU 的部分
  int cpu_result = diverge_cpu(INPUT_X, INPUT_Y, 256);

  printf("GPU vs CPU: %d, %d\n", h_c[0], cpu_result);

  cudaFree(d_c);
  free(h_c);

  return 0;
}
