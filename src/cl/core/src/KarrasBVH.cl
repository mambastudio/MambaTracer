#define DELTA(i,j) delta(sortedMortonCodes,num_prims,i,j)

#define LEAFIDX(i) (i)
#define NODEIDX(i) (mesh.size + i)

typedef struct
{
   int bound;
   int sibling;
   int left;
   int right;
   int parent;
   int isLeaf;
   int child;

}BVHNode;

int leftShift3(int x)
{
    if (x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
    x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
    x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
    return x;
}

int encodeMortonInt(int4 scaledValue)
{
    return (leftShift3(scaledValue.x) << 2) | (leftShift3(scaledValue.y) << 1) | leftShift3(scaledValue.z);
}

int encodeMorton(float4 centroid)
{
    int mortonBits = 10;
    int mortonScale = 1 << mortonBits;  //morton scale to 10 bits maximum, this will enable
                                        //to left shift three 10 bits values into a 32 bit.
    int4 scaledValue = convert_int4( centroid * mortonScale );
    return encodeMortonInt(scaledValue);
}

int delta(global int2* sortedMortonCodes, int num_prims, int i1, int i2)
{
    // Select left end
    int left = min(i1, i2);
    // Select right end
    int right = max(i1, i2);
    // This is to ensure the node breaks if the index is out of bounds
    if (left < 0 || right >= num_prims)
    {            
        return -1;
    }
    // Fetch Morton codes for both ends
    int left_code = sortedMortonCodes[left].x;
    int right_code = sortedMortonCodes[right].x;

    // Special handling of duplicated codes: use their indices as a fall
    return left_code != right_code ? clz(left_code ^ right_code) : (32 + clz(left ^ right));
}

int findSplit(global int2* sortedMortonCodes, int num_prims, int first, int end)
{
    // Fetch codes for both ends
    int left = first;
    int right = end;

    // Calculate the number of identical bits from higher end
    int num_identical = DELTA(left, right);

    do
    {
        // Proposed split
        int new_split = (right + left) / 2;

        // If it has more equal leading bits than left and right accept it
        if (DELTA(left, new_split) > num_identical)
            left = new_split;            
        else            
            right = new_split;            
    }
    while (right > left + 1);
    
    return left;
}

int2 findSpan(global int2* sortedMortonCodes, int num_prims, int idx)
{
    // Find the direction of the range
    int d = sign((float)(DELTA(idx, idx+1) - DELTA(idx, idx-1)));

    // Find minimum number of bits for the break on the other side
    int delta_min = DELTA(idx, idx-d);
    
    // Search conservative far end
    int lmax = 2;
    while (DELTA(idx,idx + lmax * d) > delta_min)
        lmax *= 2;
    
    // Search back to find exact bound
    // with binary search
    int l = 0;
    int t = lmax;
    do
    {
        t /= 2;
        if(DELTA(idx, idx + (l + t)*d) > delta_min)
        {
            l = l + t;
        }
    }
    while (t > 1);
            
    // Pack span 
    int2 span;
    span.x = min(idx, idx + l*d);
    span.y = max(idx, idx + l*d);

    return span;
}

__kernel void calculateMorton(global float4* points, global Face* faces, global int* size, global int2* mortonPrimitive)
{
     int id= get_global_id( 0 );
     TriangleMesh mesh = {points, faces, size[0]};
     
     //get bounding box (id), get center of bounding box, calcu
     int morton = encodeMorton(getCenterOfBoundingBox(getBoundingBox(mesh, id)));
     
     mortonPrimitive[id].x = morton;
     mortonPrimitive[id].y = id;
}

__kernel void test(global BVHNode* nodes, global int* nodeSize)
{
      printInt(nodeSize[0], true);
      for(int i = 0; i<23; i++)
      {

      }
}

__kernel void emitHierarchy(global float4* points, global Face* faces, global int* size, //mesh
                            global int2* sortedMortonCodes,                              //sorted mortons
                            global BVHNode* nodes,                                       //bvh nodes (nodes + leaves)
                            global BoundingBox* bounds)                                  //bounds (nodes + leaves)
{
      int id= get_global_id( 0 );
      TriangleMesh mesh = {points, faces, size[0]};
      
      int leafIndex = LEAFIDX(id);
      int nodeIndex = NODEIDX(id);
      
      //Init leaf
      if(id < mesh.size)
      {
         nodes[leafIndex].isLeaf = 1;
         nodes[leafIndex].child = sortedMortonCodes[leafIndex].y;
         nodes[leafIndex].bound = leafIndex;
         bounds[leafIndex] = getBoundingBox(mesh, nodes[leafIndex].child);
      }
      
      //Set internal nodes
      if(id < mesh.size - 1)
      {
          // Find span occupied by the current node
          int2 range = findSpan(sortedMortonCodes, mesh.size, id);
  
          // Find split position inside the range
          int  split = findSplit(sortedMortonCodes, mesh.size, range.x, range.y);
  
          // Create child nodes if needed
          int c1idx = (split == range.x) ? LEAFIDX(split) : NODEIDX(split);
          int c2idx = (split + 1 == range.y) ? LEAFIDX(split + 1) : NODEIDX(split + 1);
  
          // Set left, right child, and init bounding box
          nodes[NODEIDX(id)].left = c1idx;
          nodes[NODEIDX(id)].right = c2idx;
          bounds[NODEIDX(id)] = getInitBoundingBox();  //printFloat(nodes[NODEIDX(id)].box.maximum.y);
  
          // Set parent of left, right child  and also siblings
          nodes[c1idx].parent = NODEIDX(id);     nodes[c1idx].sibling = c2idx;
          nodes[c2idx].parent = NODEIDX(id);     nodes[c2idx].sibling = c1idx;
       }
}

__kernel void refitBounds(global int* size, global int* flags, global BVHNode* nodes, global BoundingBox* bounds)
{
    int global_id = get_global_id(0);
    int num_prims = size[0];

    // Start from leaf nodes
    if (global_id < num_prims)
    {
        // Get my leaf index
        int idx = LEAFIDX(global_id);

        do
        {
            // Move to parent node
            idx = nodes[idx].parent;
            
            printInt(idx - num_prims, true);

            // Check node's flag
            if (atomic_cmpxchg(flags + (idx - num_prims), 0, 1) == 1)
            {
                // If the flag was 1 the second child is ready and
                // this thread calculates bbox for the node

                // Fetch kids
                int lc = nodes[idx].left;
                int rc = nodes[idx].right;
                BoundingBox b1 = bounds[lc];
                BoundingBox b2 = bounds[rc];
                
                // Calculate bounds
                BoundingBox b = unionBoundingBox(b1, b2);

                // Write bounds
                bounds[idx] = b;
                nodes[idx].bound = idx;
            }
            else
            {
                // If the flag was 0 set it to 1 and bail out.
                // The thread handling the second child will
                // handle this node.
                break;
            }
        }
        while ((idx - num_prims) != 0);
    }
}