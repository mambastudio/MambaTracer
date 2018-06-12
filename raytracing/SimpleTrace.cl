
// bvh node
typedef struct
{
   BoundingBox bounds;
   int parent, sibling, left, right, child, isLeaf;
}BVHNode;

// accel structure
typedef struct
{
   global BVHNode const* nodes;
   global int     const* objects;
}BVHAccelerator;

bool isNode(BVHAccelerator accel, int nodeId)
{
   BVHNode node = accel.nodes[nodeId];
   return !node.isLeaf;
}

bool intersectMesh(Ray* ray, Intersection* isect, TriangleMesh mesh, BVHAccelerator accel)
{
   //BVH Accelerator    intersectTriangle(ray, isect, mesh, accel.objects[node.start+i])
   int currentNode = 0;
   bool hit = false;
   
   int nodeId = 0;
   long bitstack = 0;                      //be careful when you use a 32 bit integer. For deeper hierarchy traversal may lead into an infinite loop for certain scenes
   int parentId = 0, siblingId = 0;
   
   for(;;)
   {
       while(isNode(accel, nodeId))
       {
          BVHNode node        = accel.nodes[nodeId];
          parentId            = node.parent;
          siblingId           = node.sibling;
                
          BVHNode left        = accel.nodes[node.left];
          BVHNode right       = accel.nodes[node.right];
          
          float leftT[2];
          float rightT[2];
          bool leftHit        = intersectBoundT(*ray, left.bounds, leftT);
          bool rightHit       = intersectBoundT(*ray, right.bounds, rightT);
          
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
       
       if(!isNode(accel, nodeId))
       {
          BVHNode node        = accel.nodes[nodeId];
          hit |= intersectTriangle(ray, isect, mesh, accel.objects[node.child]);
          
          //This is not in the paper.
          parentId            = node.parent;
          siblingId           = node.sibling;
       }
       
       while ((bitstack & 1) == 0)  //while skip bit in the top stack is 0 traverse up the tree
       {
          if (bitstack == 0) return hit;  //if bitstack is 0 meaning stack is empty, it is now safe to exit the tree and return hit
          nodeId = parentId;
          BVHNode node = accel.nodes[nodeId];
          parentId = node.parent;
          siblingId = node.sibling;
          bitstack >>= 1;               //pop the bit in top most part os stack by right bit shifting
       }
       nodeId = siblingId;
       bitstack ^= 1;
   }

}


__kernel void traceMesh(
    global int* imageBuffer,
    global CameraStruct* camera,
    global int* width,
    global int* height,
    global const float4* points,
    global const Face*   faces,
    global const int*    size,
    global const BVHNode* nodes,
    global const int*    objects
)
{
     int id= get_global_id( 0 );
     float2 pixel = getPixel(id, width[0], height[0]);
     TriangleMesh mesh = {points, faces, size[0]};
     BVHAccelerator accel = {nodes, objects};

     Intersection isect;
     Ray ray  = getCameraRay(pixel.x, pixel.y, width[0], height[0], camera[0]); //printFloat4(ray.d);
     
     bool hit = intersectMesh(&ray, &isect, mesh, accel);

     float4 color;
     if(hit)
     {
         color = (float4)(1, 1, 1, 1);  float coeff = fabs(dot(ray.d, isect.n));    //printFloat4(isect.n);
         color *= coeff;  color.w = 1;
     }
     else
         color = (float4)(0, 0, 0, 1);

     imageBuffer[id] = getIntARGB(color);
}