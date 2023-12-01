#include "cpu_julia.hpp"
#include <iostream>
#include <vector>
#define UPPER .3
#define LOWER .22762
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
      return 0.;
  }
  if (res < LOWER)
    return 1.;
  else if (res > UPPER)
    return 0;
  else
    return (UPPER - res) / (UPPER - LOWER);
}

void kernel(unsigned char *ptr) {
  // std::vector<float> res;
  for (int y = 0; y < DIM; y++) {
    for (int x = 0; x < DIM; x++) {
      int offset = x + y * DIM;
      float juliaValue = julia(x, y);
      // res.push_back(juliaValue);
      ptr[offset * 4 + 0] = 0;
      ptr[offset * 4 + 1] = (int)(255 * juliaValue);
      ptr[offset * 4 + 2] = (int)(255 * juliaValue);
      ptr[offset * 4 + 3] = 255;
    }
  }
  // int count0 = 0;
  // int count05 = 0;
  // int count01 = 0;
  // int count00 = 0;
  // for (float value : res) {
  //   // if (value == 0)
  //   //   count0++;
  //   // else if (value < .2276) {
  //   //   // std::cout << value << std::endl;
  //   //   count05++;
  //   // } else if (value < .2277)
  //   //   count01++;
  //   // else if (value < .23)
  //   //   count00++;
  //   if (value == 1.)
  //     count0++;
  //   else if (value < .1) {
  //     // std::cout << value << std::endl;
  //     count05++;
  //   } else if (value < 1.)
  //     count01++;
  //   else if (value < 10.)
  //     count00++;
  // }
  // std::cout << "count0: " << count0 << std::endl;
  // std::cout << "count05: " << count05 << std::endl;
  // std::cout << "count01: " << count01 << std::endl;
  // std::cout << "count00: " << count00 << std::endl;
}

int main(void) {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *ptr = bitmap.get_ptr();
  // time measuring
  struct timeval start, end;
  gettimeofday(&start, 0);
  // 57 ms
  kernel(ptr);
  gettimeofday(&end, 0);
  std::cout << "time: "
            << (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec)
            << "us" << std::endl;
  bitmap.display_and_exit();
}