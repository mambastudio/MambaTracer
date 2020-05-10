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
    int4 scaledValue = (int4)(centroid.x * mortonScale, centroid.y * mortonScale, centroid.z * mortonScale, 0);
    return encodeMortonInt(scaledValue);
}

//This can be replaced by any other prefix sum that is deemed suitable (works in local size only)
void koggeStone(__global const int* in, __global int* out, __global int* groupSum, __local int* aux)
{
     int idl  = get_local_id(0); // index in workgroup
     int idg  = get_global_id(0);
     int idgr = get_group_id(0);
     int lSize = get_local_size(0);

     aux[idl] = in[idg]; //not hardware agnostic due to varying local size capability by vendor
     barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

     for(int offset = 1; offset < lSize; offset *= 2)
     {
          private int temp;
          if(idl >= offset) temp = aux[idl - offset];
          barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
          if(idl >= offset) aux[idl] = temp + aux[idl];
          barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

     }

     if(idl == (lSize-1))
       groupSum[idgr] = aux[idl];

     if(aux[idl] > 0)
       out[idg] = aux[idl - 1];
     else
       out[idg] = aux[0];
}

//prepare leaf morton nodes
__kernel void prepareMorton(//mesh
                            global const float4* points,
                            global const float4* normals,
                            global const Face*   faces,
                            global const int*    size,

                            //bvh
                            global const BVHNode* nodes)
{
    //get thread id
    int id = get_global_id( 0 );

    TriangleMesh mesh = {points, normals, faces, size[0]};

    if(id < *size)
    {
        global BVHNode* node = nodes + id;
        BoundingBox bound    = getBoundingBox(mesh, id);
        float4 center        = getCenterOfBoundingBox(bound);

        node->mortonCode     = encodeMorton(center);
        node->child          = id;
        node->isLeaf         = true;
    }
}

//after you sort child nodes
__kernel void prepareSortedLeafs(//mesh
                                 global const float4* points,
                                 global const float4* normals,
                                 global const Face*   faces,
                                 global const int*    size,

                                 //bvh
                                 global const BVHNode* nodes,
                                 global const BoundingBox* bounds,
                                 
                                 //input 
                                 global const BVHNode* input,
                                 global const BoundingBox* inputbounds)
{
    //get thread id
    int id = get_global_id( 0 );

    TriangleMesh mesh = {points, normals, faces, size[0]};

    if(id < *size)
    {
        global BVHNode* node           = nodes + id;
        global BoundingBox* childBound = bounds + id;
        global BoundingBox* inputBound = inputbounds + id;
        
        global BVHNode* inputNode      = input + id;
        
        *childBound                    = getBoundingBox(mesh, node->child);
        node->ptr                      = id;
        node->bound                    = id;

        *inputNode                     = *node;
        *inputBound                    = *childBound;
    }
}

//nearest
__kernel void nearest(global const BoundingBox* inputbounds,
                      global const BoundingBox* outputbounds,

                      global int* nearest,
                      global const int* end,
                      global const int* radius)
{
    //get thread id
    int id = get_global_id( 0 );

    if(id < *end)
    {
        float minDistance = FLT_MAX;
        int minIndex = -1;
        BoundingBox box = inputbounds[id];
       
        //search left
        for(int neighbourIndex = id - *radius; neighbourIndex < id; ++neighbourIndex)
        {
            if(neighbourIndex < 0 || neighbourIndex>= *end)
                continue;
            BoundingBox neighbourBox = inputbounds[neighbourIndex];
            addToBoundingBox2(&neighbourBox, box);
            float distance = getBoundArea(neighbourBox);
                            
            if(minDistance > distance)
            {
                minDistance = distance;
                minIndex = neighbourIndex;
            }
        }
      
        //search right
        for(int neighbourIndex = id + 1; neighbourIndex < id + *radius + 1; ++neighbourIndex)
        {
            if(neighbourIndex < 0 || neighbourIndex >= *end)
                continue;
            BoundingBox neighbourBox = inputbounds[neighbourIndex];
            addToBoundingBox2(&neighbourBox, box);
            float distance = getBoundArea(neighbourBox);
                            
            if(minDistance > distance)
            {
                minDistance = distance;
                minIndex = neighbourIndex;
            }
        }            
        nearest[id] = minIndex;
    }
}

//merge
__kernel void merge(global BVHNode* input,
                    global BoundingBox* inputbounds,
                    global BVHNode* output,
                    global BoundingBox* outputbounds,
                    
                    //bvh
                    global BVHNode* nodes,
                    global BoundingBox* bounds,

                    global int* nearest,
                    global int* end,
                    global int* node_out_idx,

                    global int* predicate,
                    global int* localscan,
                    global int* groupsum,
                    local  int* aux)
{
    //get thread id
    int idx = get_global_id( 0 );

    if(idx < *end)
    {         
        if(nearest[nearest[idx]] == idx)
        {
            if(nearest[idx] > idx)
            {
                int left = idx;
                int right = nearest[idx];

                //setup bounds
                BVHNode node;
                node.ptr = atomic_inc(node_out_idx);
                BoundingBox bound = getInitBoundingBox();
                addToBoundingBox2(&bound, inputbounds[left]);
                addToBoundingBox2(&bound, inputbounds[right]);

                //setup nodes
                node.left = input[left].ptr; node.right = input[right].ptr;

                //update sibling and parent indices
                global BVHNode* lnode = nodes + node.left;
                global BVHNode* rnode = nodes + node.right;
                lnode->sibling = rnode->ptr; lnode->parent = node.ptr;
                rnode->sibling = lnode->ptr; rnode->parent = node.ptr;

                //add to global node and global bound
                node.bound = node.ptr;
                bounds[node.ptr] = bound;
                nodes[node.ptr] = node;

                //update out and predicate
                output[idx] = node;
                outputbounds[idx] = bound;
                predicate[idx] = 1;
            }
            else
                predicate[idx] = 0;
        }
        else
        {
            output[idx] = input[idx];
            outputbounds[idx] = inputbounds[idx];
            predicate[idx] = 1;
        }

    }
    else
        predicate[idx] = 0;
    
    //use the opportunity to do some local prefix sum
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    koggeStone(predicate, localscan, groupsum, aux);
}

//sequential prefix sum on global scale (quite fast and trivial) global = 1, local = 1
__kernel void groupPrefixSum(__global int* predicate,
                             __global int* groupSum,
                             __global int* groupPrefixSum,
                             __global int* localScan,
                             __global int* groupSize,
                             __global int* localSize,
                             __global int* compactLength)
{
      for(int i = 1; i<*groupSize; i++)
          groupPrefixSum[i] = groupPrefixSum[i-1] + groupSum[i-1];

      int groupIndex = ceil(*compactLength/(float)(*localSize)) - 1; //which group index I'm I?
      *compactLength = localScan[*compactLength - 1] + groupPrefixSum[groupIndex] + predicate[*compactLength - 1];
     //printFloat(*compactLength);
}

__kernel void compact(__global int* predicate,
                      __global int* localScan,
                      __global int* groupPrefixSum,

                      __global BVHNode*        input,
                      __global BoundingBox*    inputbounds,
                      __global BVHNode*        output,
                      __global BoundingBox*    outputbounds,

                      __global int* actualLength)
{
    int global_id             = get_global_id (0);
    int local_id              = get_local_id (0);
    int group_size            = get_local_size(0);
    int group_id              = get_group_id(0);

    if(global_id < *actualLength)
    {
      if(predicate[global_id])
      {
           int lIndex = global_id;
           int gIndex = group_id;
           int aIndex = localScan[lIndex] + groupPrefixSum[gIndex];

           output[aIndex]        = input[global_id];
           outputbounds[aIndex]  = inputbounds[global_id];
      }
    }
}

__kernel void prepareNext(global int* end,
                          global int* compactlength)
{       
    *end  = *compactlength;
}
