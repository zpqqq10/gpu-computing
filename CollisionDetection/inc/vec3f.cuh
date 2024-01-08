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

#define _USE_MATH_DEFINES
#include <math.h>
#include <ostream>
#include "forceline.h"
#include "real.h"
#include <cuda_runtime.h>

#define     GLH_ZERO                REAL(0.0)
#define     GLH_EPSILON          REAL(10e-6)
#define		GLH_EPSILON_2		REAL(10e-12)
#define     equivalent(a,b)             (((a < b + GLH_EPSILON) &&\
                                                      (a > b - GLH_EPSILON)) ? true : false)
#define GLH_LARGE_FLOAT REAL(1e18f)

template <class T>
__device__ __host__ FORCEINLINE void setMax2(T& a, const T& b)
{
	if (a < b)
	{
		a = b;
	}
}

template <class T>
__device__ __host__ FORCEINLINE void setMin2(T& a, const T& b)
{
	if (b < a)
	{
		a = b;
	}
}

__device__ __host__ inline REAL lerp(REAL a, REAL b, REAL t)
{
	return a + t*(b - a);
}

#ifdef USE_DOUBLE
__device__ __host__ inline REAL fmax(REAL a, REAL b) {
	return (a > b) ? a : b;
}

__device__ __host__ inline REAL fmin(REAL a, REAL b) {
	return (a < b) ? a : b;
}
#endif

__device__ __host__ inline bool isEqual( REAL a, REAL b, REAL tol=GLH_EPSILON )
{
    return fabs( a - b ) < tol;
}

/* This is approximately the smallest number that can be
* represented by a REAL, given its precision. */
#define ALMOST_ZERO		FLT_EPSILON

#ifndef M_PI
#define M_PI 3.14159f
#endif

#include <assert.h>

class vec2f {
public:
	union {
		struct {
		REAL x, y;
		};
		struct {
		REAL v[2];
		};
	};

	__device__ __host__ FORCEINLINE vec2f ()
	{x=0; y=0;}

	__device__ __host__ FORCEINLINE vec2f(const vec2f &v)
	{
		x = v.x;
		y = v.y;
	}

	__device__ __host__ FORCEINLINE vec2f(const REAL *v)
	{
		x = v[0];
		y = v[1];
	}

	__device__ __host__ FORCEINLINE vec2f(REAL x, REAL y)
	{
		this->x = x;
		this->y = y;
	}

	// cross product
	__device__ __host__ FORCEINLINE REAL cross(const vec2f &vec) const
	{
		return x*vec.y - y*vec.x;
	}

	__device__ __host__ FORCEINLINE REAL dot(const vec2f &vec) const {
		return x*vec.x + y*vec.y;
	}

	__device__ __host__ FORCEINLINE REAL operator [] ( int i ) const {return v[i];}
	__device__ __host__ FORCEINLINE REAL &operator [] (int i) { return v[i]; }

	__device__ __host__ FORCEINLINE vec2f operator- (const vec2f &v) const
	{
		return vec2f(x - v.x, y - v.y);
	}

	
};

class vec3f {
public:
	union {
		struct {
		REAL x, y, z;
		};
		struct {
		REAL v[3];
		};
	};

	__device__ __host__ FORCEINLINE vec3f ()
	{x=0; y=0; z=0;}

	__device__ __host__ FORCEINLINE vec3f(const vec3f &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	__device__ __host__ FORCEINLINE vec3f(const REAL *v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	__device__ __host__ FORCEINLINE vec3f(REAL x, REAL y, REAL z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__device__ __host__ FORCEINLINE REAL operator [] ( int i ) const {return v[i];}
	__device__ __host__ FORCEINLINE REAL &operator [] (int i) { return v[i]; }

	__device__ __host__ FORCEINLINE vec3f &operator += (const vec3f &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__device__ __host__ FORCEINLINE vec3f &operator -= (const vec3f &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__device__ __host__ FORCEINLINE vec3f &operator *= (REAL t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	__device__ __host__ FORCEINLINE vec3f &operator /= (REAL t) {
		x /= t;
		y /= t;
		z /= t;
		return *this;
	}

	__device__ __host__ FORCEINLINE void negate() {
		x = -x;
		y = -y;
		z = -z;
	}

	__device__ __host__ FORCEINLINE vec3f absolute() const
	{
		return vec3f(fabs(x), fabs(y), fabs(z));
	}

	__device__ __host__ FORCEINLINE vec3f operator - () const {
		return vec3f(-x, -y, -z);
	}

	__device__ __host__ FORCEINLINE vec3f operator+ (const vec3f &v) const
	{
		return vec3f(x+v.x, y+v.y, z+v.z);
	}

	__device__ __host__ FORCEINLINE vec3f operator- (const vec3f &v) const
	{
		return vec3f(x-v.x, y-v.y, z-v.z);
	}

	__device__ __host__ FORCEINLINE vec3f operator *(REAL t) const
	{
		return vec3f(x*t, y*t, z*t);
	}

	__device__ __host__ FORCEINLINE vec3f operator /(REAL t) const
	{
		return vec3f(x/t, y/t, z/t);
	}

     // cross product
     __device__ __host__ FORCEINLINE const vec3f cross(const vec3f &vec) const
     {
          return vec3f(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
     }

	 __device__ __host__ FORCEINLINE REAL dot(const vec3f &vec) const {
		 return x*vec.x+y*vec.y+z*vec.z;
	 }

	 __device__ __host__ FORCEINLINE void normalize() 
	 { 
		 REAL sum = x*x+y*y+z*z;
		 if (sum > GLH_EPSILON_2) {
			 REAL base = REAL(1.0/sqrt(sum));
			 x *= base;
			 y *= base;
			 z *= base;
		 }
	 }

	 __device__ __host__ FORCEINLINE REAL length() const {
		 return REAL(sqrt(x*x + y*y + z*z));
	 }

	 __device__ __host__ FORCEINLINE vec3f getUnit() const {
		 return (*this)/length();
	 }

	__device__ __host__ FORCEINLINE bool isUnit() const {
		return isEqual( squareLength(), 1.f );
	}

    //! max(|x|,|y|,|z|)
	__device__ __host__ FORCEINLINE REAL infinityNorm() const
	{
		return fmax(fmax( fabs(x), fabs(y) ), fabs(z));
	}

	__device__ __host__ FORCEINLINE vec3f & set_value( const REAL &vx, const REAL &vy, const REAL &vz)
	{ x = vx; y = vy; z = vz; return *this; }

	__device__ __host__ FORCEINLINE bool equal_abs(const vec3f &other) {
		return x == other.x && y == other.y && z == other.z;
	}

	__device__ __host__ FORCEINLINE REAL squareLength() const {
		return x*x+y*y+z*z;
	}

	__device__ __host__ FORCEINLINE REAL length2() const {
		return x * x + y * y + z * z;
	}

	__device__ __host__ FORCEINLINE vec3f  dot3(const vec3f& v0, const vec3f& v1, const vec3f& v2) const
	{
		return vec3f(dot(v0), dot(v1), dot(v2));
	}

	/**@brief Set each element to the max of the current values and the values of another btVector3
 * @param other The other btVector3 to compare with
 */
	__device__ __host__ FORCEINLINE void	setMax(const vec3f& other)
	{
		setMax2(x, other.x);
		setMax2(y, other.y);
		setMax2(z, other.z);
	}

	/**@brief Set each element to the min of the current values and the values of another btVector3
 * @param other The other btVector3 to compare with
 */
	__device__ __host__ FORCEINLINE void	setMin(const vec3f& other)
	{
		setMin2(x, other.x);
		setMin2(y, other.y);
		setMin2(z, other.z);
	}

	__device__ __host__ static vec3f zero() {
		return vec3f(0.f, 0.f, 0.f);
	}

    //! Named constructor: retrieve vector for nth axis
	__device__ __host__ static vec3f axis( int n ) {
		assert( n < 3 );
		switch( n ) {
			case 0: {
				return xAxis();
			}
			case 1: {
				return yAxis();
			}
			case 2: {
				return zAxis();
			}
		}
		return vec3f();
	}

    //! Named constructor: retrieve vector for x axis
	__device__ __host__ static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
    //! Named constructor: retrieve vector for y axis
	__device__ __host__ static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
    //! Named constructor: retrieve vector for z axis
	__device__ __host__ static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

};

__device__ __host__ inline vec3f operator * (REAL t, const vec3f &v) {
	return vec3f(v.x*t, v.y*t, v.z*t);
}

__device__ __host__ inline vec3f interp(const vec3f &a, const vec3f &b, REAL t)
{
	return a*(1-t)+b*t;
}

__device__ __host__ inline vec3f vinterp(const vec3f &a, const vec3f &b, REAL t)
{
	return a*t+b*(1-t);
}

__device__ __host__ inline vec3f interp(const vec3f &a, const vec3f &b, const vec3f &c, REAL u, REAL v, REAL w)
{
	return a*u+b*v+c*w;
}

__device__ __host__ inline REAL clamp(REAL f, REAL a, REAL b)
{
	return fmax(a, fmin(f, b));
}

__device__ __host__ inline REAL vdistance(const vec3f &a, const vec3f &b)
{
	return (a-b).length();
}


inline std::ostream& operator<<( std::ostream&os, const vec3f &v ) {
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	return os;
}

#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)


__device__ __host__ FORCEINLINE void
vmin(vec3f &a, const vec3f &b)
{
	a.set_value(
		fmin(a[0], b[0]),
		fmin(a[1], b[1]),
		fmin(a[2], b[2]));
}

__device__ __host__ FORCEINLINE void
vmax(vec3f &a, const vec3f &b)
{
	a.set_value(
		fmax(a[0], b[0]),
		fmax(a[1], b[1]),
		fmax(a[2], b[2]));
}

__device__ __host__ FORCEINLINE vec3f lerp(const vec3f &a, const vec3f &b, REAL t)
{
	return a + t*(b - a);
}


/**@brief Return the elementwise product of two vectors */
__device__ __host__ FORCEINLINE vec3f operator*(const vec3f& v1, const vec3f& v2)
{
	return vec3f(
		v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z);
}
