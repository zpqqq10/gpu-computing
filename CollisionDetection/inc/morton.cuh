#pragma once

#include <cuda_runtime.h>
#include "definitions.h"
#include "aabb.cuh"
#include "tri3f.cuh"


__device__ __host__ FORCEINLINE morton expandBits(morton v)
{
#ifdef MORTON_CODE_64
    v = (v | (v << 32)) & 0xFFFF00000000FFFFull;
    v = (v | (v << 16)) & 0x00FF0000FF0000FFull;
    v = (v | (v << 8)) & 0xF00F00F00F00F00Full;
    v = (v | (v << 4)) & 0x30C30C30C30C30C3ull;
    v = (v | (v << 2)) & 0x9249249249249249ull;

#else
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

#endif
    return v;
}

//将质心坐标norm到[0,1]
__device__ __host__ FORCEINLINE REAL Norm(REAL element, REAL minval, REAL maxval)
{
    element = (element - minval)/(maxval - minval);

    return element;
}

__device__ __host__ FORCEINLINE morton morton3D(REAL x, REAL y, REAL z)
{     

#ifdef MORTON_CODE_64
    x = min(max(x * 1048576.0f, 0.0f), 1048575.0f);
    y = min(max(y * 1048576.0f, 0.0f), 1048575.0f);
    z = min(max(z * 1048576.0f, 0.0f), 1048575.0f);
#else
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
#endif

    morton xx = expandBits((morton)x);
    morton yy = expandBits((morton)y);
    morton zz = expandBits((morton)z);

    return xx * 4 + yy * 2 + zz;

}

// calculating moton code
__global__ void calculate_morton_kernel(
                            tri3f *triangles, 
                            const BOX &bbox, 
                            unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    /*
    norm the centriod into [0,1]
    calculate the morton code 
    */ 
    double norm_cx = Norm(triangles[i]._center.x, bbox._min.x, bbox._max.x);
    double norm_cy = Norm(triangles[i]._center.y, bbox._min.y, bbox._max.y);
    double norm_cz = Norm(triangles[i]._center.z, bbox._min.z, bbox._max.z);

    triangles[i].morton = morton3D(norm_cx, norm_cy, norm_cz);
    
}

