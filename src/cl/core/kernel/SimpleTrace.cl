#define NODEROOT() (mesh.size)

/*
  while(true)
  {
     generateCameraRays(camera, rays, isects, width, height);
     intersectScene(rays, isects, atomic_count, 
                    points, faces, size, 
                    nodes, bounds);
     updateShadeImage(imageBuffer, width, height, isects);
  }

*/

int getMaterial(int data)
{
    return data & 0xFFFF;
}

int getGroup(int data)
{
    return (data >> 16) & 0xFFFF;
}

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
    
    //get global ray
    global Ray* r = rays + id;

    //get global camera ray
    getGlobalCameraRay(r, camera, pixel.x, pixel.y, width[0], height[0]);
}

__kernel void intersectPrimitives(
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
    //get thread id
    int id = get_global_id( 0 );

    //get ray, create both isect and mesh
    global Ray* ray = rays + id;
    global Intersection* isect = isects + id;
    TriangleMesh mesh = {points, faces, size[0]};

    //intersect and update hit and update isect
    int hit = intersectGlobal(ray, isect, mesh, nodes, bounds);
    isect->hit = hit;

    //probably put a global memory fence here
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void updateShadeImage(
    global int* imageBuffer,
    global int* width,
    global int* height,
    global Intersection* isects
)
{
    int id= get_global_id( 0 );
    float4 color = (float4)(0, 0, 0, 1);
    
    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        float coeff = fabs(dot(isect->d, isect->n));
        color = (float4)(1, 1, 1, 1);
        color.xyz *= coeff;
    }
    //update
    imageBuffer[id] = getIntARGB(color);
}

__kernel void groupBufferPass(
    global Intersection* isects,
    global int* groupBuffer
)
{
    int id= get_global_id( 0 );
    
    global Intersection* isect = isects + id;
    
    if(isect->hit)
        groupBuffer[id] = 0;//getGroup(isects[id].mat);
    else
        groupBuffer[id] = -1;
}