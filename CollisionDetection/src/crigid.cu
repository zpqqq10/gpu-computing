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

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include "morton.cuh"
#include "crigid.cuh"
#include "aabb.cuh"
#include "helper_cuda.h"

#include <set>
using namespace std;

// for fopen
#pragma warning(disable: 4996)

// returning the normal of triangle (v1, v2, v3)
inline vec3f update(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2-v1);
	return s.cross(v3-v1);
}

inline vec3f
update(tri3f &tri, vec3f *vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

inline vec3f
update(tri3f &tri, thrust::host_vector<vec3f>& vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

kmesh::kmesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl)
: _tris(tris, tris + numTri), _vtxs(vtxs, vtxs + numVtx)
{
		_num_vtx = numVtx;
		_num_tri = numTri;

		_fnrms = nullptr;
		_nrms = nullptr;
		_dl = -1;

		updateBxs();
		// calculate mortons 
		thrust::device_vector<morton> d_mortons(numTri);
		thrust::device_vector<BOX> d_bbox(1);
		d_bbox[0] = _bx;
		d_tris = _tris;
		d_vtxs = _vtxs;
		calculate_morton_kernel<<<(numTri + 32 - 1) / 32, 32>>>(
			thrust::raw_pointer_cast(d_mortons.data()),
			thrust::raw_pointer_cast(d_tris.data()),
			thrust::raw_pointer_cast(d_vtxs.data()),
			thrust::raw_pointer_cast(d_bbox.data()),
			numTri);
		cudaDeviceSynchronize();
		getLastCudaError("calculate mortons failed");

		// sort triangles and bboxes
		thrust::device_vector<int> sorted_indices(numTri);
		d_bxs = _bxs;
		thrust::device_vector<BOX> d_bxs_sorted(numTri);
		thrust::device_vector<tri3f> d_tris_sorted(numTri);
		// multiple argsort needed here, thus sorting an index vector at first
    	thrust::sequence(sorted_indices.begin(), sorted_indices.end());
		thrust::sort_by_key(d_mortons.begin(), d_mortons.end(), sorted_indices.begin());
		thrust::gather(sorted_indices.begin(), sorted_indices.end(), d_tris.begin(), d_tris_sorted.begin());
		thrust::gather(sorted_indices.begin(), sorted_indices.end(), d_bxs.begin(), d_bxs_sorted.begin());
		getLastCudaError("sort mortons and triangles and bboxes failed");
		d_tris = d_tris_sorted;
		d_bxs = d_bxs_sorted;
		_tris = d_tris_sorted;
		_bxs = d_bxs_sorted;

		updateNrms();
		updateDL(cyl, -1);
	}

void kmesh::updateNrms()
{
	if (_fnrms == nullptr)
		_fnrms = new vec3f[_num_tri];

	if (_nrms == nullptr)
		_nrms = new vec3f[_num_vtx];

	for (unsigned int i = 0; i < _num_tri; i++) {
		vec3f n = ::update(_tris[i], _vtxs);
		n.normalize();
		_fnrms[i] = n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i] = vec3f::zero();

	for (unsigned int i=0; i<_num_tri; i++) {
		vec3f& n = _fnrms[i];
		_nrms[_tris[i].id0()] += n;
		_nrms[_tris[i].id1()] += n;
		_nrms[_tris[i].id2()] += n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i].normalize();
}

void kmesh::updateBxs() {
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

void kmesh::display(bool cyl, int level)
{
	if (_dl == -1)
		updateDL(cyl, level);

	glCallList(_dl);
	displayDynamic(cyl, level);
}

void kmesh::displayStatic(bool cyl, int level)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
#ifdef USE_DOUBLE
	glVertexPointer(3, GL_DOUBLE, sizeof(REAL) * 3, _vtxs.data());
	glNormalPointer(GL_DOUBLE, sizeof(REAL) * 3, _nrms);
#else
	glVertexPointer(3, GL_FLOAT, sizeof(REAL) * 3, _vtxs.data());
	glNormalPointer(GL_FLOAT, sizeof(REAL) * 3, _nrms);
#endif

	glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris.data());

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

}

void kmesh::displayDynamic(bool cyl, int level)
{
}

void kmesh::destroyDL()
{
	if (_dl != -1)
		glDeleteLists(_dl, 1);
}

void kmesh::updateDL(bool cyl, int level)
{
	if (_dl != -1)
		destroyDL();

	_dl = glGenLists(1);
	glNewList(_dl, GL_COMPILE);
	displayStatic(cyl, level);
	glEndList();
}

void drawOther()
{
}

void crigid::checkCollision(crigid *rb, std::vector<id_pair>&pairs)
{
	crigid* ra = this;
	const transf& trfA = ra->getTrf();
	const transf& trfB = rb->getTrf();
	//const transf trfA2B = trfB.inverse() * trfA;

	ra->getMesh()->collide(rb->getMesh(), trfA, trfB, pairs);
}

void crigid::checkCollision(crigid *rb, thrust::host_vector<int, INT_PINNED>&face0, thrust::host_vector<int, INT_PINNED>&face1)
{
	crigid* ra = this;
	const transf& trfA = ra->getTrf();
	const transf& trfB = rb->getTrf();
	//const transf trfA2B = trfB.inverse() * trfA;

	ra->getMesh()->collide(rb->getMesh(), trfA, trfB, face0, face1);
}


void beginDraw(BOX& bx)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	vec3f pt = bx.center();
	REAL len = bx.height() + bx.depth() + bx.width();
	REAL sc = 6.0 / len;

	//glRotatef(-90, 0, 0, 1);
	glScalef(sc, sc, sc);
	glTranslatef(-pt.x, -pt.y, -pt.z);
}

void endDraw()
{
	glPopMatrix();
}


void drawRigid(crigid* r, bool cyl, int level, vec3f& off)
{
	glPushMatrix();
	glTranslated(off.x, off.y, off.z);

	//glLoadIdentity();
	matrix3f R = r->getRot();
	vec3f T = r->getOffset();
	std::vector<GLdouble> Twc = { R(0, 0), R(1, 0), R(2, 0), 0.,
							  R(0, 1), R(1, 1), R(2, 1), 0.,
							  R(0, 2), R(1, 2), R(2, 2), 0.,
							  T[0], T[1], T[2], 1. };

	glMultMatrixd(Twc.data());
	r->getMesh()->display(cyl, level);
	glPopMatrix();
}

void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs)
{
	if (pairs.size() == 0)
		return;

	transf& t0 = r0->getWorldTransform();
	transf& t1 = r1->getWorldTransform();
	kmesh* km0 = r0->getMesh();
	kmesh* km1 = r1->getMesh();

	glDisable(GL_LIGHTING);
	{
		glColor3f(1, 0, 0);
		glBegin(GL_TRIANGLES);
		for (auto t : pairs) {
			unsigned int fid0, fid1;
			t.get(fid0, fid1);

			vec3f v0, v1, v2;
			km0->getTriangleVtxs(fid0, v0, v1, v2);
			vec3f p0 = t0.getVertex(v0);
			vec3f p1 = t0.getVertex(v1);
			vec3f p2 = t0.getVertex(v2);

#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
			glVertex3dv(p2.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
			glVertex3fv(p2.v);
#endif

			km1->getTriangleVtxs(fid1, v0, v1, v2);
			p0 = t1.getVertex(v0);
			p1 = t1.getVertex(v1);
			p2 = t1.getVertex(v2);
#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
			glVertex3dv(p2.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
			glVertex3fv(p2.v);
#endif

		}
		glEnd();
	}

	glEnable(GL_LIGHTING);
}

void drawCDPair(crigid* r0, crigid* r1, thrust::host_vector<int, INT_PINNED>& faces0, thrust::host_vector<int, INT_PINNED>& faces1)
{
	if (faces0.size() == 0)
		return;

	transf& t0 = r0->getWorldTransform();
	transf& t1 = r1->getWorldTransform();
	kmesh* km0 = r0->getMesh();
	kmesh* km1 = r1->getMesh();

	glDisable(GL_LIGHTING);
	{
		glColor3f(1, 0, 0);
		glBegin(GL_TRIANGLES);
		// draw collided faces in r0
		for (auto fid0 : faces0) {
			vec3f v0, v1, v2;
			km0->getTriangleVtxs(fid0, v0, v1, v2);
			vec3f p0 = t0.getVertex(v0);
			vec3f p1 = t0.getVertex(v1);
			vec3f p2 = t0.getVertex(v2);

#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
			glVertex3dv(p2.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
			glVertex3fv(p2.v);
#endif

		}
		// draw collided faces in r1
		for (auto fid1 : faces1) {
			vec3f v0, v1, v2;
			km1->getTriangleVtxs(fid1, v0, v1, v2);
			vec3f p0 = t1.getVertex(v0);
			vec3f p1 = t1.getVertex(v1);
			vec3f p2 = t1.getVertex(v2);
#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
			glVertex3dv(p2.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
			glVertex3fv(p2.v);
#endif

		}
		glEnd();
	}

	glEnable(GL_LIGHTING);
}