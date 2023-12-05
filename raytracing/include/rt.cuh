#pragma once

#include "CPUBitmap.hpp"
#include "book.h"
#include "cpu_anim.h"
#include "cuda.h"
#include <curand_kernel.h>
#include <iostream>

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define dev_rnd(x, s) (curand(&s) % 10000 * 1.0 / 10000 * x) // rand() on device
// one for x, one for y, one for z, one for radius
#define SPHERE_ANIMATION_WIDTH 4
// how many ticks to change the animation
#define FRAME_LENGTH 10
// number of spheres
#define SPHERES 2000
#define AMPLITUDE 10
#define CONSTANTMEMORY
// #define SHAREMEMORY

struct Sphere {
  float r, b, g;
  float radius;
  float x, y, z;
  int idx; // index, to identify a sphere
  __device__ float hit(float ox, float oy, float *n, int *offset) {
    float current_x = offset[0] + x;
    float current_y = offset[1] + y;
    float current_z = offset[2] + z;
    float current_r = offset[3] + radius;

    float dx = ox - current_x;
    float dy = oy - current_y;
    if (dx * dx + dy * dy < current_r * current_r) {
      float dz = sqrtf(current_r * current_r - dx * dx - dy * dy);
      *n = dz / sqrtf(current_r * current_r);
      return dz + z;
    }

    return -INF;
  }
};

// globals needed by the update routine
struct DataBlock {
  unsigned char *dev_bitmap;
  Sphere *s;
  CPUAnimBitmap *bitmap; // for animation
  int *animationOffset;  // offset for animation, x, y, z and radius
  // for recording time
  cudaEvent_t start, stop;
};

// allocate cuda memory for DataBlock
void allocateDataBlock(DataBlock *d) {
  HANDLE_ERROR(cudaEventCreate(&d->start));
  HANDLE_ERROR(cudaEventCreate(&d->stop));
  // allocate memory on the GPU for the output bitmap
  HANDLE_ERROR(cudaMalloc((void **)&d->dev_bitmap, d->bitmap->image_size()));
#ifdef CONSTANTMEMORY
  std::cout << "Using constant memory" << std::endl;
#else
  // allocate memory for the Sphere dataset
  HANDLE_ERROR(cudaMalloc((void **)&d->s, sizeof(Sphere) * SPHERES));
#endif
  // allocate memory for animation offset
  HANDLE_ERROR(cudaMalloc((void **)&d->animationOffset,
                          sizeof(int) * SPHERE_ANIMATION_WIDTH));
}

// update offset of each sphere
__global__ void updateSphereShifts(int *animationOffset, int seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state;
  curand_init(seed, tid, 0, &state);
  // while (i < SPHERES) {
  animationOffset[SPHERE_ANIMATION_WIDTH * tid] =
      (int)dev_rnd(AMPLITUDE, state); // x
  animationOffset[SPHERE_ANIMATION_WIDTH * tid + 1] =
      (int)dev_rnd(AMPLITUDE, state); // y
  animationOffset[SPHERE_ANIMATION_WIDTH * tid + 2] =
      (int)dev_rnd(AMPLITUDE, state); // z
  animationOffset[SPHERE_ANIMATION_WIDTH * tid + 3] =
      (int)dev_rnd(AMPLITUDE, state); // radius

  // i += SPHERE_BLOCK;
  // }
}
