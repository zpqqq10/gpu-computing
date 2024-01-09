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

#pragma once

#include "vec3f.cuh"
#include "transf.cuh"
#include <float.h>
#include <vector>
#include <cuda_runtime.h>

#define BOX aabb

class aabb {
public:
	__device__ __host__ FORCEINLINE void init() {
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	vec3f _min;
	vec3f _max;

	__device__ __host__ FORCEINLINE aabb() {
		init();
	}

	__device__ __host__ FORCEINLINE aabb(const vec3f &v) {
		_min = _max = v;
	}

	__device__ __host__ FORCEINLINE aabb(const vec3f &a, const vec3f &b) {
		_min = a;
		_max = a;
		vmin(_min, b);
		vmax(_max, b);
	}

	__device__ __host__ FORCEINLINE bool overlaps(const aabb& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;
		if (_min[2] > b._max[2]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;
		if (_max[2] < b._min[2]) return false;

		return true;
	}

	__device__ __host__ FORCEINLINE bool overlaps(const aabb& b, REAL tol) const
	{
		aabb aa = *this;
		aabb bb = b;

		aa.enlarge(tol);
		bb.enlarge(tol);
		return aa.overlaps(bb);
	}

	__device__ __host__ FORCEINLINE bool overlaps(const aabb &b, aabb &ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min = vec3f(
			fmax(_min[0], b._min[0]),
			fmax(_min[1], b._min[1]),
			fmax(_min[2], b._min[2]));

		ret._max = vec3f(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]),
			fmin(_max[2], b._max[2]));

		return true;
	}

	__device__ __host__ FORCEINLINE bool inside(const vec3f &p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;
		if (p[2] < _min[2] || p[2] > _max[2]) return false;

		return true;
	}

	__device__ __host__ FORCEINLINE aabb &operator += (const vec3f &p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	__device__ __host__ FORCEINLINE aabb &operator += (const aabb &b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	__device__ __host__ FORCEINLINE aabb operator + (const aabb &v) const
	{
		aabb rt(*this); return rt += v;
	}

	__device__ __host__ FORCEINLINE REAL width()  const { return _max[0] - _min[0]; }
	__device__ __host__ FORCEINLINE REAL height() const { return _max[1] - _min[1]; }
	__device__ __host__ FORCEINLINE REAL depth()  const { return _max[2] - _min[2]; }
	__device__ __host__ FORCEINLINE vec3f center() const { return (_min + _max)*REAL(0.5); }
	__device__ __host__ FORCEINLINE REAL volume() const { return width()*height()*depth(); }


	__device__ __host__ FORCEINLINE bool empty() const {
		return _max[0] < _min[0];
	}

	__device__ __host__ FORCEINLINE void enlarge(REAL thickness) {
		_max += vec3f(thickness, thickness, thickness);
		_min -= vec3f(thickness, thickness, thickness);
	}

	__device__ __host__ FORCEINLINE const vec3f &getMax() const { return _max; }
	__device__ __host__ FORCEINLINE const vec3f &getMin() const { return _min; }
	__device__ __host__ FORCEINLINE void setMax(vec3f& v) { _max = v; }
	__device__ __host__ FORCEINLINE void setMin(vec3f& v) { _min = v; }

	__device__ __host__ void getCorners(std::vector<vec3f> &crns) {
		crns.push_back(_max);
		crns.push_back(vec3f(_max.x, _max.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _max.z));
		crns.push_back(_min);
		crns.push_back(vec3f(_min.x, _max.y, _min.z));
		crns.push_back(vec3f(_min.x, _max.y, _max.z));
		crns.push_back(vec3f(_min.x, _min.y, _max.z));
	}

	//! Apply a transform to an AABB
	__device__ __host__ FORCEINLINE void applyTransform(const transf &trans)
	{
		vec3f c =center();
		vec3f extends = _max - c;
		// Compute new center
		c = trans(c);

		vec3f textends = extends.dot3(trans.getBasis().getRow(0).absolute(),
			trans.getBasis().getRow(1).absolute(),
			trans.getBasis().getRow(2).absolute());

		_min = c - textends;
		_max = c + textends;
	}

	//! Gets the extend and center
	__device__ __host__ FORCEINLINE void getCenterExtend(vec3f & center, vec3f & extend)  const
	{
		center = (_min + _max) * 0.5f;
		extend = _max - center;
	}

	void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	void visualize();
};


// bounding sphere
class Bsphere {
public:
	vec3f _center;
	REAL _radius;


	__device__ __host__ FORCEINLINE void init() {
		_center = vec3f(0, 0, 0);
		_radius = 0;
	}

	__device__ __host__ FORCEINLINE void init(const vec3f& p0, const vec3f& p1, const vec3f& p2) {
		vec3f sum = p0 + p1 + p2;
		_center = sum / 3.0;
		_radius = fmax(
				fmax((_center - p0).length(), (_center - p1).length()), 
				(_center - p0).length()) + 1e-6;
	}

	// https://zhuanlan.zhihu.com/p/457319176 welzl
	__device__ __host__ FORCEINLINE void init_min(const vec3f& p0, const vec3f& p1, const vec3f& p2) {
		vec3f a = p1 - p0;
		vec3f b = p2 - p0;
		vec3f c = p2 - p1;
		vec3f a_x_b = a.cross(b);
		REAL al = a.length();
		REAL bl = b.length();
		REAL cl = c.length();
		_radius = (al * bl * cl) / (2 * a_x_b.length()) + 1e-6;
		_center = p0 + (bl * bl * a_x_b.cross(a) + al * al * b.cross(a_x_b)) / (2 * a_x_b.length() * a_x_b.length());

	}

	__device__ __host__ FORCEINLINE Bsphere() {
		init();
	}

	__device__ __host__ FORCEINLINE Bsphere(const Bsphere &b) {
		_center = b._center;
		_radius = b._radius;
	}

	__device__ __host__ FORCEINLINE bool overlaps(const Bsphere& b) const
	{
		// REAL d = (_center - b._center).length();
		// return (d < (_radius + b._radius));
		REAL d = (_center - b._center).length2();
		return (d < ((_radius + b._radius) * (_radius + b._radius)));
	}

	// get center
	__device__ __host__ FORCEINLINE vec3f getCenter() const
	{
		return _center;
	}

	// get radius
	__device__ __host__ FORCEINLINE REAL getRadius() const
	{
		return _radius;
	}

	__device__ __host__ FORCEINLINE void translate(const vec3f &v)
	{
		_center += v;
	}

	// set center
	__device__ __host__ FORCEINLINE void setCenter(const vec3f &v)
	{
		_center = v;
	}

	__device__ __host__ FORCEINLINE void setRadius(REAL r)
	{
		_radius = r;
	}


	FORCEINLINE bool inside(const vec3f &p) const
	{
		return (_center - p).length() < _radius;
	}

};

inline std::ostream& operator<<( std::ostream&os, const Bsphere &v ) {
	os << "[" << v._center << ", " << v._radius << "]";
	return os;
}