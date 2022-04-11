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

__kernel void IntersectPrimitives(
    global Ray* rays,
    global Intersection* isects,
    global int* count,

    //mesh
    global const float4* points,
    global const float2* uvs,
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
    TriangleMesh mesh = {points, uvs, normals, faces, size[0]};
    
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

          float2 uv, texuv;
          
          bool hasTexUV = hasUV(mesh, childIndex);
          
          if(hasUV(mesh, childIndex))
          {
              float2 uv1 = getUV1(mesh, childIndex);
              float2 uv2 = getUV2(mesh, childIndex);
              float2 uv3 = getUV3(mesh, childIndex);
              texuv = triangleBarycentricsFromUVMesh(p, p1, p2, p3, uv1, uv2, uv3);

          }
         
          uv = triangleBarycentrics(p, p1, p2, p3);

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

              //the order of uvw matters
              n = n1 * (1 - tuv[1] - tuv[2]) + n2 * tuv[1] + n3 * tuv[2];
              if(dot(n, ray->d)> 0.00001f)
              {
                  n  = -n;
                  
                  //reverse normal if necessary for use below in the shadow terminator
                  n1 = -n1;
                  n2 = -n2;
                  n3 = -n3;
              }

              //if you don't normalize, bsdf, specifically anisotropic, will have issues
              n  = normalize(n);
              

              //Hacking the shadow terminator by Johannes Hanika
              float4 tmpu = makeFloat4(0, 0, 0, 0);
              float4 tmpv = makeFloat4(0, 0, 0, 0);
              float4 tmpw = makeFloat4(0, 0, 0, 0);

              tmpu.xyz = p.xyz - p1.xyz;
              tmpv.xyz = p.xyz - p2.xyz;
              tmpw.xyz = p.xyz - p3.xyz;
              
              float dotu = min(0.0f, dot(tmpu.xyz, normalize(n1.xyz)));
              float dotv = min(0.0f, dot(tmpv.xyz, normalize(n2.xyz)));
              float dotw = min(0.0f, dot(tmpw.xyz, normalize(n3.xyz)));
              
              tmpu.xyz -= dotu*normalize(n1.xyz);
              tmpv.xyz -= dotv*normalize(n2.xyz);
              tmpw.xyz -= dotw*normalize(n3.xyz);

              //why the order of uvw matters (has to match with above)
              p = p + (1 - tuv[1] - tuv[2])*tmpu + tuv[1]*tmpv + tuv[2]*tmpw;

          }
          else
          {
              n  = getNormal(p1, p2, p3);
              if(dot(n, ray->d)>0.00001f)
                  n  = -n;
          }
              
          float4 ng = getNormal(p1, p2, p3);
          if(dot(ng, ray->d)>0.00001f)
              ng  = -ng;
              
          //set values
          isect->p  = p;
          isect->n  = n;
          isect->ng = ng;
          isect->d  = ray->d;
          
          if(hasTexUV)
            isect->uv = texuv;
          else
            isect->uv = uv;
            
          isect->id = childIndex;
          isect->mat = getMaterial(mesh.faces[childIndex].mat);  //because face - mat is encoded to include both group and material, as such, extract material index
          isect->hit = hit; //update hit status
      }
      else
      {
        isect->hit = MISS_MARKER;
        isect->mat = -1;
      }
    }
    else
    {
      isect->hit = MISS_MARKER;
      isect->mat = -1;
    }
}

bool testOcclusion(
    global Ray* ray,

    //mesh
    TriangleMesh mesh,

    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds
)
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
                return true;
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

__kernel void intersectOcclusion(
    global Ray* rays,
    global int* hits,
    global int* count,

    //mesh
    global const float4* points,
    global const float2* uvs,
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
    TriangleMesh mesh = {points, uvs, normals, faces, size[0]};

    if(id < *count)
      //intersect
      *hit = intersectMesh(ray, &childIndex, mesh, nodes, bounds, true);
}
