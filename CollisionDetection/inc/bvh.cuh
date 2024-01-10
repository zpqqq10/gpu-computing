#pragma once

#include <cuda_runtime.h>
#include "morton.cuh"
#include "tri3f.cuh"
#include "vec3f.cuh"
#include "vector_functions.h"

class BVHNode {
public:
    unsigned int idx;               // index of node

    bool isLeaf;                    // if leaf node
	BVHNode *parent;                // parent node
    BVHNode *left, *right;          // left/right child node, only internal nodes have
    BOX box;                        // AABB bounding box
    // index? pointer?
    tri3f *triangle;             // corresponding triangle, only leaf nodes have

    __host__ __device__ BVHNode() { 
        parent = NULL; 
        left = right = NULL;
        triangle = NULL;
    }
};


/*find the split index*/
__device__ int findSplit(morton* mortons,
                        const unsigned int first,
                        const unsigned int last);


/*find the range of objects in the internal node*/
__device__ vec2i determineRange(morton* mortons,
                                const unsigned int num_tris, // originally numObjects, a triangle is exactly an object here
                                const unsigned int i);

/*generate a BVH tree by iteration*/
__global__ void build_bvh_kernel(
                                morton* mortons,
                                BVHNode *LeafNodes, 
                                BVHNode *InternalNodes,
                                const unsigned int num_tris);
