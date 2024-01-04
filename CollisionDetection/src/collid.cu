//**************************************************************************************
//  Copyright (C) 2022 - 2024, Min Tang (tang_m@zju.edu.cn)
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#include <set>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <stdio.h>

using namespace std;
#include "mat3f.cuh"
#include "box.h"
#include "crigid.h"
#include "helper_cuda.h"


inline double fmax(double a, double b, double c)
{
	double t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

inline double fmin(double a, double b, double c)
{
	double t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

inline int project3(const vec3f& ax,
	const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

inline int project6(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);
	double Q1 = ax.dot(q1);
	double Q2 = ax.dot(q2);
	double Q3 = ax.dot(q3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);
	double mx2 = fmax(Q1, Q2, Q3);
	double mn2 = fmin(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles
// 分离轴定理
bool triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
	vec3f p1;			// default to be (0, 0, 0)
	// relative coordinates, relative to p1(0, 0, 0)
	vec3f p2 = P2 - P1;
	vec3f p3 = P3 - P1;
	vec3f q1 = Q1 - P1;
	vec3f q2 = Q2 - P1;
	vec3f q3 = Q3 - P1;

	// edge of triangle 1
	vec3f e1 = p2 - p1;
	vec3f e2 = p3 - p2;
	vec3f e3 = p1 - p3;

	// edge of triangle 2
	vec3f f1 = q2 - q1;
	vec3f f2 = q3 - q2;
	vec3f f3 = q1 - q3;

	// normal of triangle 1
	vec3f n1 = e1.cross(e2);
	// normal of triangle 2
	vec3f m1 = f1.cross(f2);

	// axis
	vec3f g1 = e1.cross(n1);
	vec3f g2 = e2.cross(n1);
	vec3f g3 = e3.cross(n1);

	vec3f h1 = f1.cross(m1);
	vec3f h2 = f2.cross(m1);
	vec3f h3 = f3.cross(m1);

	vec3f ef11 = e1.cross(f1);
	vec3f ef12 = e1.cross(f2);
	vec3f ef13 = e1.cross(f3);
	vec3f ef21 = e2.cross(f1);
	vec3f ef22 = e2.cross(f2);
	vec3f ef23 = e2.cross(f3);
	vec3f ef31 = e3.cross(f1);
	vec3f ef32 = e3.cross(f2);
	vec3f ef33 = e3.cross(f3);

	// now begin the series of tests
	if (!project3(n1, q1, q2, q3)) return false;
	if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

	return true;
}

// void kmesh::collide(const kmesh* other, const transf& t0, const transf &t1, std::vector<id_pair>& rets)
// {
// 	// thrust::device_vector<int> D(10, 1);
// 	// thrust::host_vector<int> E(10, 1);
// 	// check all the triangles pair by pair
// 	for (unsigned int i = 0; i < _num_tri; i++) {

// 		for (unsigned int j = 0; j < other->_num_tri; j++) {
// 			vec3f v0, v1, v2;
// 			this->getTriangleVtxs(i, v0, v1, v2);
// 			vec3f p0 = t0.getVertex(v0);
// 			vec3f p1 = t0.getVertex(v1);
// 			vec3f p2 = t0.getVertex(v2);

// 			other->getTriangleVtxs(j, v0, v1, v2);
// 			vec3f q0 = t1.getVertex(v0);
// 			vec3f q1 = t1.getVertex(v1);
// 			vec3f q2 = t1.getVertex(v2);

// 			if (triContact(p0, p1, p2, q0, q1, q2))
// 				rets.push_back(id_pair(i, j, false));
// 		}
// 	}
// }

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles
// 分离轴定理
// 输出的bool存到一个数组里，数组的长度即为网格的三角形数目，值为0/1，代表这个面片是否发生了碰撞（和哪个面片发生了碰撞则不太重要）
// __device__ bool triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
// {
// 	vec3f p1;			// default to be (0, 0, 0)
// 	// relative coordinates, relative to p1(0, 0, 0)
// 	vec3f p2 = P2 - P1;
// 	vec3f p3 = P3 - P1;
// 	vec3f q1 = Q1 - P1;
// 	vec3f q2 = Q2 - P1;
// 	vec3f q3 = Q3 - P1;

// 	// edge of triangle 1
// 	vec3f e1 = p2 - p1;
// 	vec3f e2 = p3 - p2;
// 	vec3f e3 = p1 - p3;

// 	// edge of triangle 2
// 	vec3f f1 = q2 - q1;
// 	vec3f f2 = q3 - q2;
// 	vec3f f3 = q1 - q3;

// 	// normal of triangle 1
// 	vec3f n1 = e1.cross(e2);
// 	// normal of triangle 2
// 	vec3f m1 = f1.cross(f2);

// 	// axis
// 	vec3f g1 = e1.cross(n1);
// 	vec3f g2 = e2.cross(n1);
// 	vec3f g3 = e3.cross(n1);

// 	vec3f h1 = f1.cross(m1);
// 	vec3f h2 = f2.cross(m1);
// 	vec3f h3 = f3.cross(m1);

// 	vec3f ef11 = e1.cross(f1);
// 	vec3f ef12 = e1.cross(f2);
// 	vec3f ef13 = e1.cross(f3);
// 	vec3f ef21 = e2.cross(f1);
// 	vec3f ef22 = e2.cross(f2);
// 	vec3f ef23 = e2.cross(f3);
// 	vec3f ef31 = e3.cross(f1);
// 	vec3f ef32 = e3.cross(f2);
// 	vec3f ef33 = e3.cross(f3);

// 	// now begin the series of tests
// 	if (!project3(n1, q1, q2, q3)) return false;
// 	if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

// 	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
// 	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

// 	return true;
// }

__device__ FORCEINLINE vec3f getTransformedVertex(
	const vec3f& _off, const matrix3f& _trf, 
	const vec3f& _vtx
){
	return _off + _trf * _vtx;
}

// __global__ void kernel_collide(tri3f *mesh0_tris, tri3f *mesh1_tris, 
// 							   const vec3f *mesh0_vtxs, const vec3f *mesh1_vtxs,
// 							   const unsigned int mesh0_num_tri, const unsigned int mesh1_num_tri,
// 							   const unsigned int mesh0_num_vtx, const unsigned int mesh1_num_vtx,
// 							   const transf& t0, const transf& t1, 
// 							   bool *result)
// {
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	int j = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (i >= mesh0_num_tri || j >= mesh1_num_tri) return;

// 	vec3f v0, v1, v2;
// 	tri3f& face0 = mesh0_tris[i];
// 	vec3f p0 = getTransformedVertex(t0._off, t0._trf, mesh0_vtxs[face0.id0()]);
// 	vec3f p1 = getTransformedVertex(t0._off, t0._trf, mesh0_vtxs[face0.id1()]);
// 	vec3f p2 = getTransformedVertex(t0._off, t0._trf, mesh0_vtxs[face0.id2()]);

// 	tri3f& face1 = mesh1_tris[j];
// 	vec3f q0 = getTransformedVertex(t1._off, t1._trf, mesh1_vtxs[face1.id0()]);
// 	vec3f q1 = getTransformedVertex(t1._off, t1._trf, mesh1_vtxs[face1.id1()]);
// 	vec3f q2 = getTransformedVertex(t1._off, t1._trf, mesh1_vtxs[face1.id2()]);

// 	result[i * mesh1_num_tri + j] = triContact(p0, p1, p2, q0, q1, q2);
// }

void kmesh::collide(const kmesh* other, const transf& t0, const transf &t1, std::vector<id_pair>& rets)
{
	// in total: this->_num_tri * other->_num_tri
	// warp: 32 * 32
	// check all the triangles pair by pair
	for (unsigned int i = 0; i < _num_tri; i++) {
		printf("checking %d of %d...\n", i, _num_tri);

		for (unsigned int j = 0; j < other->_num_tri; j++) {
			vec3f v0, v1, v2;
			this->getTriangleVtxs(i, v0, v1, v2);
			vec3f p0 = t0.getVertex(v0);
			vec3f p1 = t0.getVertex(v1);
			vec3f p2 = t0.getVertex(v2);

			other->getTriangleVtxs(j, v0, v1, v2);
			vec3f q0 = t1.getVertex(v0);
			vec3f q1 = t1.getVertex(v1);
			vec3f q2 = t1.getVertex(v2);

			if (triContact(p0, p1, p2, q0, q1, q2))
				rets.push_back(id_pair(i, j, false));
		}
	}
}