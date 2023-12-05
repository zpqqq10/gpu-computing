#include "rt.cuh"

#define DIM 1024
#ifdef CONSTANTMEMORY
// must be a global variable and cannot be a argument
__constant__ Sphere s[SPHERES];

// for every pixel, compute whether it is inside a sphere and is hited by a ray
__global__ void kernel(int *animationOffset, unsigned char *ptr) {
#else
__global__ void kernel(Sphere *s, int *animationOffset, unsigned char *ptr) {
#endif

#ifdef SHAREMEMORY
  __shared__ int sharedOffset[SPHERES * SPHERE_ANIMATION_WIDTH];

  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int thread_per_block = blockDim.x * blockDim.y;
  int cur_thread_id = thread_id;

  while (cur_thread_id < SPHERES) {
    sharedOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id] =
        animationOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id];
    sharedOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 1] =
        animationOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 1];
    sharedOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 2] =
        animationOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 2];
    sharedOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 3] =
        animationOffset[SPHERE_ANIMATION_WIDTH * cur_thread_id + 3];
    cur_thread_id += thread_per_block;
  }

  __syncthreads();
#endif
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - DIM / 2);
  float oy = (y - DIM / 2);

  float r = 0, g = 0, b = 0;
  float maxz = -INF;
  for (int i = 0; i < SPHERES; i++) {
    float n;
    float t =
#ifdef SHAREMEMORY
        s[i].hit(ox, oy, &n, sharedOffset + i * SPHERE_ANIMATION_WIDTH);
#else
        s[i].hit(ox, oy, &n, animationOffset + i * SPHERE_ANIMATION_WIDTH);
#endif
    if (t > maxz) {
      float fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = t;
    }
  }

  ptr[offset * 4 + 0] = (int)(r * 255);
  ptr[offset * 4 + 1] = (int)(g * 255);
  ptr[offset * 4 + 2] = (int)(b * 255);
  ptr[offset * 4 + 3] = 255;
}

#ifdef CONSTANTMEMORY
void allocateSpheresOnConstant(int len) {
  std::cout << "Using constant memory" << std::endl;
  // allocate temp memory, initialize it, copy to constant
  // memory on the GPU, then free our temp memory
  Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * len);
  for (int i = 0; i < len; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500; // x,y,z 在初始化时已经中心化过了
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(50.0f) + 10;
    temp_s[i].idx = i;
  }
  HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * len));

  // free temp_s on host
  free(temp_s);
}
#else
void allocateSpheres(Sphere *target, int len) {
  // allocate temp memory, initialize it, copy to constant
  // memory on the GPU, then free our temp memory
  Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * len);
  for (int i = 0; i < len; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500; // x,y,z 在初始化时已经中心化过了
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(50.0f) + 10;
    temp_s[i].idx = i;
  }
  HANDLE_ERROR(
      cudaMemcpy(target, temp_s, sizeof(Sphere) * len, cudaMemcpyHostToDevice));

  // free temp_s on host
  free(temp_s);
}
#endif

// called between frames
void animate(DataBlock *d, int ticks) {
  // change every ten frames
  if (ticks % FRAME_LENGTH == 0) {
    HANDLE_ERROR(cudaEventRecord(d->start, 0));

    dim3 grids(DIM / 32, DIM / 32);
    dim3 threads(32, 32);

    updateSphereShifts<<<SPHERES, 1>>>(d->animationOffset, ticks);
#ifdef CONSTANTMEMORY
    kernel<<<grids, threads>>>(d->animationOffset, d->dev_bitmap);
#else
    kernel<<<grids, threads>>>(d->s, d->animationOffset, d->dev_bitmap);
#endif

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
                            d->bitmap->image_size(), cudaMemcpyDeviceToHost));

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    printf("Time for a frame:  %3.1f ms\n", elapsedTime);
  }
}

// called when exiting
void exitAnimation(DataBlock *d) {
  HANDLE_ERROR(cudaFree(d->dev_bitmap));

#ifdef CONSTANTMEMORY
  ;
#else
  HANDLE_ERROR(cudaFree(d->s));
#endif
  HANDLE_ERROR(cudaFree(d->animationOffset));
  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {

  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;

  allocateDataBlock(&data);
#ifdef CONSTANTMEMORY
  allocateSpheresOnConstant(SPHERES);
#else
  allocateSpheres(data.s, SPHERES);
#endif

  // display
  bitmap.anim_and_exit((void (*)(void *, int))animate,
                       (void (*)(void *))exitAnimation);
}
