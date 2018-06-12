#define NODEROOT() (mesh.size)

__kernel void traceMesh(
    global int* imageBuffer,
    global CameraStruct* camera,
    global int* width,
    global int* height,
    global const float4* points,
    global const Face*   faces,
    global const int*    size,
    global const BVHNode* nodes,
    global const BoundingBox* bounds)
{
     int id= get_global_id( 0 );
     float2 pixel = getPixel(id, width[0], height[0]);
     TriangleMesh mesh = {points, faces, size[0]};

     Intersection isect;
     Ray ray  = getCameraRay(pixel.x, pixel.y, width[0], height[0], camera[0]); //printFloat4(ray.d);
     

     bool hit = intersect(&ray, &isect, mesh, nodes, bounds);

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