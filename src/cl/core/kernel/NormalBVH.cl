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

bool intersectMesh(global Ray* ray, int* childIndex, TriangleMesh mesh, global BVHNode* nodes, global BoundingBox* bounds, bool bailout)
{
     //BVH Accelerator
     bool hit = false;

     int nodeId = 0;
     long bitstack = 0;                      //be careful when you use a 32 bit integer. For deeper hierarchy traversal may lead into an infinite loop for certain scenes, hence 64 bit long is preferable
     int parentId = 0, siblingId = 0;

     for(;;)
     {
        while(!nodes[nodeId].isLeaf)
        { 
            BVHNode node        = nodes[nodeId];
            parentId            = node.parent;
            siblingId           = node.sibling;
                  
            BVHNode left        = nodes[node.left];
            BVHNode right       = nodes[node.right];
            
            float leftT[2];
            float rightT[2];
            bool leftHit        = intersectBoundT(*ray, bounds[left.bound], leftT);
            bool rightHit       = intersectBoundT(*ray, bounds[right.bound], rightT);

            if(!leftHit && !rightHit)
                 break;
              
            bitstack <<= 1; //push 0 bit into bitstack to skip the sibling later
                  
            if(leftHit && rightHit)
            {                    
                nodeId = (rightT[0] < leftT[0]) ? node.right : node.left;
                bitstack |= 1; //change skip code to 1 to traverse the sibling later
            }
            else
            {
                nodeId = leftHit ? node.left : node.right;                   
            }
        }

        if(nodes[nodeId].isLeaf)
        {
            BVHNode node        = nodes[nodeId];

            float4 p1 = getP1(mesh, node.child);
            float4 p2 = getP2(mesh, node.child);
            float4 p3 = getP3(mesh, node.child);
            
            float t = fastTriangleIntersection(*ray, p1, p2, p3);
            
            if(isInside(*ray, t))
            {
                if(bailout)
                   return true;

                hit = true;
                ray->tMax = t;
                *childIndex = node.child;
            }

            //This is not in the paper.
            parentId            = node.parent;
            siblingId           = node.sibling;
        }
        
        while ((bitstack & 1) == 0)  //while skip bit in the top stack is 0 traverse up the tree
        {
            if (bitstack == 0) return hit;  //if bitstack is 0 meaning stack is empty, it is now safe to exit the tree and return hit
            nodeId = parentId;
            BVHNode node = nodes[nodeId];
            parentId = node.parent;
            siblingId = node.sibling;
            bitstack >>= 1;               //pop the bit in top most part os stack by right bit shifting
        }
        nodeId = siblingId;
        bitstack ^= 1;
     }
}

__kernel void intersectPrimitives(
    global Ray* rays,
    global Intersection* isects,
    global int* count,

    //mesh
    global const float4* points,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,

    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds
)
{
    //get thread id
    int id = get_global_id( 0 );
    int childIndex;

    //get ray, create both isect and mesh
    global Ray* ray = rays + id;
    global Intersection* isect = isects + id;
    TriangleMesh mesh = {points, normals, faces, size[0]};
    
    if(id < *count)
    {
      //intersect
      bool hit = intersectMesh(ray, &childIndex, mesh, nodes, bounds, false);
      if(hit)
      {
          float4 p1 = getP1(mesh, childIndex);
          float4 p2 = getP2(mesh, childIndex);
          float4 p3 = getP3(mesh, childIndex);
          float4 p  = getPoint(*ray, ray->tMax);

          float2 uv = triangleBarycentrics(p, p1, p2, p3);
          float tuv[3];
          tuv[0] = ray->tMax;
          tuv[1] = uv.x;
          tuv[2] = uv.y;

          float4 n;

          if(hasNormals(mesh, childIndex))
          {
              float4 n1 = getN1(mesh, childIndex);
              float4 n2 = getN2(mesh, childIndex);
              float4 n3 = getN3(mesh, childIndex);

              n = n1 * (1 - tuv[1] - tuv[2]) + n2 * tuv[1] + n3 * tuv[2];
          }
          else
              n  = getNormal(p1, p2, p3);
    
          //set values
          isect->p = p;
          isect->n = n;
          isect->d = ray->d;
          isect->id = childIndex;
          isect->mat = getMaterial(mesh.faces[childIndex].mat);  //because face - mat is encoded to include both group and material, as such, extract material index
        
          //update hit status and what pixel it represent
          isect->pixel = ray->pixel;
          isect->hit = hit;
      }
      else
      {
        isect->hit = MISS_MARKER;
        isect->mat = -1;
      }
    }
}

__kernel void intersectOcclusion(
    global Ray* rays,
    global int* hits,
    global int* count,

    //mesh
    global const float4* points,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,

    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds
)
{
    //get thread id
    int id = get_global_id( 0 );
    int childIndex;

    //get ray, create both isect and mesh
    global Ray* ray = rays + id;
    global int* hit = hits + id;
    TriangleMesh mesh = {points, normals, faces, size[0]};

    if(id < *count)
      //intersect
      *hit = intersectMesh(ray, &childIndex, mesh, nodes, bounds, true);
}
