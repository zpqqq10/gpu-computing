#pragma once

#include <cuda_runtime.h>
#include "morton.cuh"
#include "tri3f.cuh"
#include "vec3f.cuh"

class BVHNode {
public:
    unsigned int idx;               // index of node

    bool isLeaf;                    // if leaf node
	BVHNode *parent;                   // parent node
    BVHNode *childA, *childB;          // left/right child node, only internal nodes have
    Box box;                        // AABB bounding box
    // index? pointer?
    tri3f *triangle;             // corresponding triangle, only leaf nodes have

    __host__ __device__ BVHNode() { 
        parent = NULL; 
        childA = childB = NULL;
        triangle = NULL;
    }
};

#define delta(i,j,keys,n) ((j >= 0 && j < n) ? __clzll(keys[i] ^ keys[j]) : -1) 


/*find the split index*/
__device__ int findSplit(morton* mortons,
                        const unsigned int first,
                        const unsigned int last)
{
    morton first_morton = mortons[first];
    morton last_morton = mortons[last];

    // if same, return the middle position of them
    if (first_morton == last_morton)
    {
        return (first + last) >> 1;
    }

    int commonPrefix = __clzll(first_morton ^ last_morton);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    unsigned int split = first; // initial guess
    unsigned int step = last - first;

    do
    {
        step = (step + 1) >> 1;      // exponential decrease
        unsigned int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            morton split_morton = mortons[newSplit];
            int splitPrefix = __clzll(first_morton ^ split_morton);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}


/*find the range of objects in the internal node*/
__device__ vec2i determineRange(morton* mortons,
                                const unsigned int numTris, // originally numObjects, a triangle is exactly an object here
                                const unsigned int i)
{
    int d = (delta(i, i + 1, mortons, numTris) - delta(i, i - 1, mortons, numTris)) >= 0 ? 1 : -1;
    int delta_min = delta(i, i - d, mortons, numTris);
    int mlen = 2;
    //return Range(100, mlen);
    while (delta(i, i + mlen * d, mortons, numTris) > delta_min) {
        mlen <<= 1;
    }

    int l = 0;
    for (int t = mlen >> 1; t >= 1; t >>= 1) {
        if (delta(i, i + (l + t) * d, mortons, numTris) > delta_min) {
            l += t;
        }
    }
    unsigned int j = i + l * d;

    // the former is smaller and the latter is bigger
    return vec2i(min(i, j), max(i, j));
}

/*generate a BVH tree by iteration*/
__global__ void build_bvh_kernel(
                                morton* mortons,
                                BVHNode *LeafNodes, 
                                BVHNode *InternalNodes,
                                const unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    
    //Construct interal Nodes
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    vec2i range = determineRange(mortons, num_tris, i);
    int first = range.x;                        
    int last = range.y;

    // Determine where to split the range.
    int split = findSplit(mortons, first, last);

    // Select leftChild.

    BVHNode *left;
    if (split == first)
        left = &LeafNodes[split];
    else
        left = &InternalNodes[split];

    // Select rightChild.

    BVHNode *right;
    if (split + 1 == last)
        right = &LeafNodes[split + 1];
    else
        right = &InternalNodes[split + 1];

    // Record parent-child relationships.
    InternalNodes[i].LeftChild = left;
    InternalNodes[i].RightChild = right;
    left->parent = &InternalNodes[i];
    right->parent = &InternalNodes[i];
}

/*assign index of leafnodes*/
__global__ void AssignIndexOfLeafNodes(BVHNode *LeafNodes,
                                        tri3f* triangles,
                                        unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    
    //Construct Leaf Nodes
    LeafNodes[i].ObjectIndex = triangles[i].Index;
    LeafNodes[i].SortedObjectIndex = i;
}