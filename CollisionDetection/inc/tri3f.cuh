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
#include "definitions.h"
#include "vec3f.cuh"
#include <cuda_runtime.h>


// triangle, storing three vertex indices
class tri3f {
public:
	unsigned int _ids[3];
	//  kmesh::displayStatic set the size of tri3f

	__device__ __host__ FORCEINLINE tri3f() {
		// become the largest unsigned int
		_ids[0] = _ids[1] = _ids[2] = -1;
	}

	__device__ __host__ FORCEINLINE tri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	__device__ __host__ FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_ids[0] = id0;
		_ids[1] = id1;
		_ids[2] = id2;
	}

	__device__ __host__ FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	__device__ __host__ FORCEINLINE unsigned int id0() {return _ids[0];}
	__device__ __host__ FORCEINLINE unsigned int id1() {return _ids[1];}
	__device__ __host__ FORCEINLINE unsigned int id2() {return _ids[2];}
	__device__ __host__ FORCEINLINE unsigned int id0() const {return _ids[0];}
	__device__ __host__ FORCEINLINE unsigned int id1() const {return _ids[1];}
	__device__ __host__ FORCEINLINE unsigned int id2() const {return _ids[2];}
	// std::swap is not suppotred in device
	FORCEINLINE void reverse() {std::swap(_ids[0], _ids[2]);}
};

inline std::ostream& operator<<( std::ostream&os, const tri3f &tri ) {
	os << "(" << tri._ids[0]<< ", " << tri._ids[1] << ", " << tri._ids[2] << ")" << std::endl;
	return os;
}