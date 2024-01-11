#pragma once

#include <cuda_runtime.h>
#include "definitions.h"
#include "aabb.cuh"
#include "tri3f.cuh"
#include "vec3f.cuh"

class BVHNode {
public:
    unsigned int idx;               // index of node
    unsigned int calculated = 0;       // whether the bbox has been calculated

    bool isLeaf;                    // if leaf node
	BVHNode *parent;                // parent node
    BVHNode *left, *right;          // left/right child node, only internal nodes have
    BOX box;                        // bounding box
    BOX obox;                       // original bounding box, to be used with transform
    tri3f *tri;                    // triangles in the leaf node, only leaf nodes have

    __host__ __device__ BVHNode() { 
        parent = NULL; 
        left = right = NULL;
        isLeaf = false;
        tri = NULL;
    }
};

// generate a BVH tree
__global__ void build_bvh_kernel(
                                morton* mortons,
                                BVHNode *LeafNodes, 
                                BVHNode *InternalNodes,
                                const unsigned int num_tris);

// init leaf nodes or internal nodes
__global__ void init_bvhleaves_kernel(BVHNode* nodes, tri3f *tris, const unsigned int num_tris);

// calculate bboxes bottom-up
__global__ void cal_bboxes_kernel(BVHNode* leaves, BOX* leaf_bboxes, unsigned int num_tris);

// calculating moton code
__global__ void calculate_morton_kernel(
                            morton *mortons,
                            const tri3f *triangles, 
                            const vec3f *vertices,
                            const BOX *bbox,
                            unsigned int num_tris);