#include "CPUBitmap.hpp"
#include <iostream>
#include <sys/time.h>

#define DIM 1024.0
#define ROUNDS 200

struct cuComplex {
  float r;
  float i;
  cuComplex(float a, float b) : r(a), i(b) {}
  float magnitude2(void) { return r * r + i * i; }
  cuComplex operator*(const cuComplex &a) {
    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
  }
  cuComplex operator+(const cuComplex &a) {
    return cuComplex(r + a.r, i + a.i);
  }
};