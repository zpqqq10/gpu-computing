#include "bvh.cuh"

/************************** for morton **************************/ 
__device__ __host__ FORCEINLINE morton expandBits(morton v)
{
    v = (v | (v << 32)) & 0xFFFF00000000FFFFull;
    v = (v | (v << 16)) & 0x00FF0000FF0000FFull;
    v = (v | (v << 8)) & 0xF00F00F00F00F00Full;
    v = (v | (v << 4)) & 0x30C30C30C30C30C3ull;
    v = (v | (v << 2)) & 0x9249249249249249ull;

    return v;
}

// normalize the element into [0,1]
__device__ __host__ FORCEINLINE REAL Norm(REAL element, REAL minval, REAL maxval)
{
    element = (element - minval)/(maxval - minval);

    return element;
}

__device__ __host__ FORCEINLINE morton morton3D(REAL x, REAL y, REAL z)
{     
    x = min(max(x * 1048576.0f, (REAL)0.0f), (REAL)1048575.0f);
    y = min(max(y * 1048576.0f, (REAL)0.0f), (REAL)1048575.0f);
    z = min(max(z * 1048576.0f, (REAL)0.0f), (REAL)1048575.0f);

    morton xx = expandBits((morton)x);
    morton yy = expandBits((morton)y);
    morton zz = expandBits((morton)z);

    return xx * 4 + yy * 2 + zz;
}

// calculating moton code
__global__ void calculate_morton_kernel(
                            morton *mortons,
                            const tri3f *triangles, 
                            const vec3f *vertices,
                            const BOX *bbox,
                            unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    // normalize the center
    vec3f center = (vertices[triangles[i].id0()] + vertices[triangles[i].id1()] + vertices[triangles[i].id2()]) / 3.0f;
    REAL norm_cx = Norm(center.x, bbox->_min.x, bbox->_max.x);
    REAL norm_cy = Norm(center.y, bbox->_min.y, bbox->_max.y);
    REAL norm_cz = Norm(center.z, bbox->_min.z, bbox->_max.z);

    mortons[i] = morton3D(norm_cx, norm_cy, norm_cz);
}


/************************** for bvh **************************/ 

#define delta(i,j,keys,n) ((j >= 0 && j < n) ? __clzll(keys[i] ^ keys[j]) : -1) 

// find the split index
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

// find the range of objects in the internal node
__device__ vec2i determineRange(morton* mortons,
                                const unsigned int num_tris, // originally numObjects, a triangle is exactly an object here
                                const unsigned int i)
{
    int d = (delta(i, i + 1, mortons, num_tris) - delta(i, i - 1, mortons, num_tris)) >= 0 ? 1 : -1;
    int delta_min = delta(i, i - d, mortons, num_tris);
    int mlen = 2;
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

// generate a BVH tree
__global__ void build_bvh_kernel(
                                morton* mortons,
                                BVHNode *leaves, 
                                BVHNode *internals,
                                const unsigned int num_tris)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;
    
    // Construct interal Nodes
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
        left = &leaves[split];
    else
        left = &internals[split];

    // Select rightChild.
    BVHNode *right;
    if (split + 1 == last)
        right = &leaves[split + 1];
    else
        right = &internals[split + 1];

    // Record parent-child relationships.
    internals[i].left = left;
    internals[i].right = right;
    left->parent = &internals[i];
    right->parent = &internals[i];
}

// triangles here have been sorted according to morton
__global__ void init_bvhleaves_kernel(BVHNode* nodes, tri3f *tris, const unsigned int num_tris)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;

    nodes[i].left = NULL;
    nodes[i].right = NULL;
    nodes[i].parent = NULL;
    nodes[i].isLeaf = true;
    nodes[i].tri = &tris[i];
    nodes[i].idx = i;
}

// calculate bboxes bottom-up
__global__ void cal_bboxes_kernel(BVHNode* leaves, BOX* leaf_bboxes, unsigned int num_tris) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_tris) return;

    // set bbox of leaves
    leaves[i].box = leaf_bboxes[i];
    leaves[i].obox = leaf_bboxes[i];
    BVHNode* current = leaves[i].parent;

    // bottom-up calculation
    while (current != NULL) {
        // ensure the two children have been calculated
        if (atomicAdd(&(current->calculated), 1) == 0) {
            break; 
        }
        else {
            BOX newbox;
            newbox += current->left->box;
            newbox += current->right->box;
            current->box.setMax(newbox._max);
            current->box.setMin(newbox._min);
            current->obox.setMax(newbox._max);
            current->obox.setMin(newbox._min);
            current = current->parent;
        }
    }

}