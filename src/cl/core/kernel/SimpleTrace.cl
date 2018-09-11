#define NODEROOT() (mesh.size)

__kernel void generateCameraRays(
    global CameraStruct* camera,
    global Ray* rays,
    global Intersection* isects,
    global int* width,
    global int* height)
{
    //global id and pixel making
    int id= get_global_id( 0 );
    float2 pixel = getPixel(id, width[0], height[0]);

    //create camera ray and set it active
    Ray ray = getCameraRay(pixel.x, pixel.y, width[0], height[0], camera[0]);
    setRayActive(&ray, true);

    //assign ray and pixel
    rays[id] = ray;
    isects[id].pixel = pixel;
}

__kernel void intersectMain(
    global Ray* rays,
    global Intersection* isects,
    global int* count,

    //mesh
    global const float4* points,
    global const Face*   faces,
    global const int*    size,

    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds
)
{
    //global id and mesh creation
    int id= get_global_id( 0 );
    TriangleMesh mesh = {points, faces, size[0]};

    //if ray not active just close the thread
    Ray ray = rays[id];
    if(!isRayActive(ray))
       return;
    
    //only accept hit intersection
    Intersection isect;
    bool hit = intersect(&ray, &isect, mesh, nodes, bounds);
    if(!hit)
       return;

    //align intersection
    int dst = atom_inc(count);
    isects[dst] = isect;
}

__kernel void updateShadeImage(
    global int* imageBuffer,
    global int* width,
    global int* height,
    global Intersection* isects
)
{
    //get global id and pixel
    int id= get_global_id( 0 );
    float2 pixel = getPixel(id, width[0], height[0]);
    
    //get intersect
    Intersection isect = isects[id];


    //get shade color
    float4 color;
    if(isect.hit)
    {
        color = (float4)(1, 1, 1, 1);  float coeff = fabs(dot(isect.d, isect.n));    //printFloat4(isect.n);
        color *= coeff;  color.w = 1;
    }
    else
        color = (float4)(0, 0, 0, 1);
    
    //update
    imageBuffer[id] = getIntARGB(color);
}

__kernel void traceMesh(
    global int* imageBuffer,
    global CameraStruct* camera,
    global int* width,
    global int* height,
    
    //mesh
    global const float4* points,
    global const Face*   faces,
    global const int*    size,
    
    //bvh
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