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

#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory.h> // thrust::system::cuda::universal_host_pinned_memory_resource;
// using pinned_allocator_type = thrust::mr::stateless_resource_allocator<T, thrust::universal_host_pinned_memory_resource>;
// #include <thrust/system/cuda/experimental/pinned_allocator.h>
// typedef thrust::cuda::experimental::pinned_allocator<T> pinned;

#include "vec3f.cuh"
#include "mat3f.cuh"
#include "transf.cuh"

#include "tri3f.cuh"
#include "aabb.cuh"
#include "pair.h"

#include <set>
#include <vector>
using namespace std;

#define INT_PINNED    thrust::mr::stateless_resource_allocator<int, thrust::universal_host_pinned_memory_resource>
#define FLOAT_PINNED  thrust::mr::stateless_resource_allocator<float, thrust::universal_host_pinned_memory_resource>
#define DOUBLE_PINNED thrust::mr::stateless_resource_allocator<double, thrust::universal_host_pinned_memory_resource>


class kmesh {
public:
	unsigned int _num_vtx;	// number of vertices
	unsigned int _num_tri;  // number of triangles
	thrust::host_vector<tri3f> _tris;			// array of triangles
	thrust::host_vector<vec3f> _vtxs;			// array of vertices
	thrust::host_vector<Bsphere> _bsphs;		// array of bounding spheres

	// thrust::host_vector<vec3f> _fnrms;			// array of face normals
	// thrust::host_vector<vec3f> _nrms;			// array of vertex normals
	thrust::host_vector<BOX> _bxs;				// bboxes of each triangles
	vec3f *_fnrms;			// array of face normals
	vec3f *_nrms;			// array of vertex normals
	BOX _bx;				// bbox of the whole mesh
	int _dl;				// display list

public:
	kmesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl): _tris(tris, tris + numTri), _vtxs(vtxs, vtxs + numVtx) {
		_num_vtx = numVtx;
		_num_tri = numTri;

		// for(unsigned int i = 0; i < numTri; i++) {
		// 	tri3f &a = _tris[i];
		// 	vec3f &p0 = _vtxs[a.id0()];
		// 	vec3f &p1 = _vtxs[a.id1()];
		// 	vec3f &p2 = _vtxs[a.id2()];

		// 	a.setCenter((p0 + p1 + p2) / 3.0f);
		// }

		// bounding sphere is slower than aabb, abandoned
		// _bsphs.resize(numTri);
		// for (unsigned int i = 0; i < numTri; i++) {
		// 	tri3f &a = _tris[i];
		// 	vec3f p0 = _vtxs[a.id0()];
		// 	vec3f p1 = _vtxs[a.id1()];
		// 	vec3f p2 = _vtxs[a.id2()];

		// 	_bsphs[i].init_min(p0, p1, p2);
		// }

		_fnrms = nullptr;
		_nrms = nullptr;
		// _bxs = nullptr;
		_dl = -1;

		updateNrms();
		updateBxs();
		updateDL(cyl, -1);
	}

	~kmesh() {

		if (_fnrms != nullptr)
			delete[] _fnrms;
		if (_nrms != nullptr)
			delete[] _nrms;
		
		destroyDL();
	}


	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	// vec3f *getVtxs() const { return _vtxs; }
	thrust::host_vector<vec3f> getVtxs() const { return _vtxs; }
	// tri3f* getTris() const { return _tris; }
	thrust::host_vector<tri3f> getTris() const { return _tris; }
	thrust::host_vector<Bsphere> getBsphs() const { return _bsphs; }
	thrust::host_vector<BOX> getBboxes() const { return _bxs; }
	vec3f* getNrms() const { return _nrms; }
	vec3f* getFNrms() const { return _fnrms; }

	// calc norms, and prepare for display ...
	void updateNrms();

	// really displaying ...
	void display(bool cyl, int);
	void displayStatic(bool cyl, int);
	void displayDynamic(bool cyl, int);

	// prepare for display
	void updateDL(bool cyl, int);
	// finish for display
	void destroyDL();

	void collide(const kmesh* other, const transf& trf, const transf &trfOther, std::vector<id_pair>& rets);
	void collide(const kmesh* other, const transf& trf, const transf &trfOther, thrust::host_vector<int, INT_PINNED>& faces0, thrust::host_vector<int, INT_PINNED>& faces1);


	void getTriangleVtxs(int fid, vec3f& v0, vec3f& v1, vec3f& v2) const
	{
		const tri3f& f = _tris[fid];
		v0 = _vtxs[f.id0()];
		v1 = _vtxs[f.id1()];
		v2 = _vtxs[f.id2()];
	}

	void updateBxs() {
		if (_bxs.size() == 0){
			_bxs.resize(_num_tri);
			// _bxs = new aabb[_num_tri];
		}

		_bx.init();

		for (unsigned int i = 0; i < _num_tri; i++) {
			tri3f &a = _tris[i];
			vec3f p0 = _vtxs[a.id0()];
			vec3f p1 = _vtxs[a.id1()];
			vec3f p2 = _vtxs[a.id2()];

			BOX bx(p0, p1);
			bx += p2;
			_bxs[i] = bx;

			_bx += bx;
		}
	}

	BOX bound() {
		return _bx;
	}


};


class crigid {
private:
	int _id;
	kmesh* _mesh;
	aabb _bx;
	transf _worldTrf;

	void updateBx() {
		_bx = _mesh->bound();
		_bx.applyTransform(_worldTrf);
	}

	//for bvh
	int _levelSt;
	int _levelNum;

public:
	crigid(kmesh* km, const transf& trf, float mass) {
		_mesh = km;
		updateBx();
		_levelSt = -1;
	}


	~crigid() {
		;
	}

	void setID(int i) { _id = i; }
	int getID() { return _id; }

	BOX bound() {
		return _bx;
	}

	kmesh* getMesh() {
		return _mesh;
	}

	const transf &getTrf() const {
		return _worldTrf;
	}

	const matrix3f &getRot() const
	{
		return _worldTrf._trf;
	}

	const vec3f &getOffset() const
	{
		return _worldTrf._off;
	}

	void updatePos(matrix3f& rt, vec3f offset = vec3f())
	{
		_worldTrf._trf = rt;
		_worldTrf._off = offset;

		updateBx();
	}

	int getVertexCount() {
		return _mesh->getNbVertices();
	}

	vec3f getVertex(int i) {
		vec3f& p = _mesh->getVtxs()[i];
		return _worldTrf.getVertex(p);
	}

	void checkCollision(crigid*, std::vector<id_pair>&);
	void checkCollision(crigid *, thrust::host_vector<int, INT_PINNED>&, thrust::host_vector<int, INT_PINNED>&);

	__forceinline transf& getWorldTransform() {
		return _worldTrf;
	}

	__forceinline void	setWorldTransform(const transf& worldTrans) {
		_worldTrf = worldTrans;
	}


};

void beginDraw(BOX &);
void endDraw();
void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs);
void drawCDPair(crigid* r0, crigid* r1, thrust::host_vector<int, INT_PINNED>& faces0, thrust::host_vector<int, INT_PINNED>& faces1);
void drawRigid(crigid*, bool cyl, int level, vec3f &);