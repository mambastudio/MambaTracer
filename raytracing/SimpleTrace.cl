
// bvh node
typedef struct
{
   BoundingBox bounds;
   int primOffset, skipIndex;
}BVHNode;

// accel structure
typedef struct
{
   global BVHNode const* nodes;
   int            const  length;
   global int     const* objects;
}BVHAccelerator;

bool isLeaf(BVHNode node)
{
   return node.skipIndex == 0;
}

bool intersectMesh(Ray* ray, Intersection* isect, TriangleMesh mesh, BVHAccelerator accel)
{
   //BVH Accelerator    intersectTriangle(ray, isect, mesh, accel.objects[node.start+i])
   int currentNode = 0;
   bool hit = false;
   
   while(currentNode < accel.length)
   {
       BVHNode  node = accel.nodes[currentNode];
       if(intersectBound(*ray, node.bounds))
       {
          if(isLeaf(node))
             hit |= intersectTriangle(ray, isect, mesh, accel.objects[node.primOffset]);
          currentNode++;
       }
       else
       {
          if(node.skipIndex > 0)
              currentNode = node.skipIndex;
          else
              currentNode++;
       }
   }

   return hit;
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
    global const int*     nodesSize,
    global const int*    objects
)
{
     int id= get_global_id( 0 );
     float2 pixel = getPixel(id, width[0], height[0]);
     TriangleMesh mesh = {points, faces, size[0]};
     BVHAccelerator accel = {nodes, nodesSize[0], objects};

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