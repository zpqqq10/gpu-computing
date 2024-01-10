#include "morton.cuh"
#include "bvh.cuh"

#define delta(i,j,keys,n) ((j >= 0 && j < n) ? __clzll(keys[i] ^ keys[j]) : -1) 

__global__ void calculate_morton_kernel(
                            morton *mortons,
                            const tri3f *triangles, 
                            const vec3f *vertices,
                            const BOX *bbox,
                            unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    /*
    norm the centriod into [0,1]
    calculate the morton code 
    */ 
    vec3f center = (vertices[triangles[i].id0()] + vertices[triangles[i].id1()] + vertices[triangles[i].id2()]) / 3.0f;
    REAL norm_cx = Norm(center.x, bbox->_min.x, bbox->_max.x);
    REAL norm_cy = Norm(center.y, bbox->_min.y, bbox->_max.y);
    REAL norm_cz = Norm(center.z, bbox->_min.z, bbox->_max.z);

    mortons[i] = morton3D(norm_cx, norm_cy, norm_cz);
}

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
                                const unsigned int num_tris, // originally numObjects, a triangle is exactly an object here
                                const unsigned int i)
{
    int d = (delta(i, i + 1, mortons, num_tris) - delta(i, i - 1, mortons, num_tris)) >= 0 ? 1 : -1;
    int delta_min = delta(i, i - d, mortons, num_tris);
    int mlen = 2;
    //return Range(100, mlen);
    while (delta(i, i + mlen * d, mortons, num_tris) > delta_min) {
        mlen <<= 1;
    }

    int l = 0;
    for (int t = mlen >> 1; t >= 1; t >>= 1) {
        if (delta(i, i + (l + t) * d, mortons, num_tris) > delta_min) {
            l += t;
        }
    }
    unsigned int j = i + l * d;

    // the former is smaller and the latter is bigger
    return vec2i(min(i, j), max(i, j));
}

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
    InternalNodes[i].left = left;
    InternalNodes[i].right = right;
    left->parent = &InternalNodes[i];
    right->parent = &InternalNodes[i];
}