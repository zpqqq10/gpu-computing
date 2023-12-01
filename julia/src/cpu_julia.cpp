#include "cpu_julia.hpp"
#include <iostream>
#include <vector>
#define UPPER .2277
#define LOWER .2276
#define RADIUS ((DIM) * (DIM) / 4.)

float julia(int x, int y) {
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
}

void kernel(unsigned char *ptr) {
  for (int y = 0; y < DIM; y++) {
    for (int x = 0; x < DIM; x++) {
      int offset = x + y * DIM;
      float juliaValue = julia(x, y);
      ptr[offset * 4 + 0] = 0;
      ptr[offset * 4 + 1] = int(255 * juliaValue);
      ptr[offset * 4 + 2] = int(255 * juliaValue);
      ptr[offset * 4 + 3] = 255;
    }
  }
}

int main(void) {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *ptr = bitmap.get_ptr();
  // time measuring
  struct timeval start, end;
  gettimeofday(&start, 0);
  // ~ 116400 ms
  kernel(ptr);
  gettimeofday(&end, 0);
  std::cout << "time: "
            << (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec)
            << "us" << std::endl;
  bitmap.display_and_exit();
}