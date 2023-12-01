#include "book.h"
#include "gpu_julia.cuh"
#include <algorithm>
#include <iostream>

#define RADIUS ((DIM) * (DIM) / 4.)

__device__ float julia(int x, int y) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
  float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
  cuComplex c(-0.413, 0.526);
  cuComplex a(jx, jy);
  int i = 0;
  float res = 0.;
  for (i = 0; i < ROUNDS; i++) {
    a = a * a + c;
    res = a.magnitude2();
    if (res > 1000.)
      return i / 30.;
  }
  res =
      ((DIM / 2 - x) * (DIM / 2 - x) + (DIM / 2 - y) * (DIM / 2 - y)) / RADIUS;
  return (1 - res);
  // return 1;
}

__global__ void kernel(unsigned char *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;
  // now calculate the value at that position
  float juliaValue = julia(x, y);
  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = (int)(255 * juliaValue);
  ptr[offset * 4 + 2] = (int)(255 * juliaValue);
  ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
  unsigned char *dev_bitmap;
};

int main(void) {
  DataBlock data;
  CPUBitmap bitmap(DIM, DIM, &data);

  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
  data.dev_bitmap = dev_bitmap;

  dim3 grid(DIM, DIM);
  // warmup
  kernel<<<grid, 1>>>(dev_bitmap);
  // time measuring
  struct timeval start, end;
  gettimeofday(&start, 0);
  kernel<<<grid, 1>>>(dev_bitmap);
  // synchronize: ~2000 us
  // no synchronization: 3us (time of calling API)
  cudaDeviceSynchronize();
  gettimeofday(&end, 0);
  std::cout << "time: "
            << (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec)
            << "us" << std::endl;

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                          cudaMemcpyDeviceToHost));

  bitmap.display_and_exit();

  HANDLE_ERROR(cudaFree(dev_bitmap));
}