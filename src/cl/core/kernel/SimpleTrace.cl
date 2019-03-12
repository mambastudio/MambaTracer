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
    bool hit = intersectGlobal(ray, isect, mesh, nodes, bounds);
    isect->hit = hit;
}

__kernel void fastShade(
    global Material* materials,
    global Intersection* isects
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //default color
    float4 color = (float4)(0, 0, 0, 1);
    float4 color1 = (float4)(1, 1, 1, 1);

    //get intersection and material
    global Intersection* isect = isects + id;
    
    if(isect->hit)
    {
        float coeff = fabs(dot(isect->d, isect->n));
        color.xyz = materials[isect->mat].diffuse.xyz;    //printlnInt(isect->mat);
        color.xyz *= coeff;
    }  
    
    isect->throughput = color;
}

__kernel void updateShadeImage(
    global int* imageBuffer,
    global int* width,
    global int* height,
    global Intersection* isects
)
{
    int id= get_global_id( 0 );
    global Intersection* isect = isects + id;

    //update
    imageBuffer[id] = getIntARGB(isect->throughput);
}

__kernel void groupBufferPass(
    global Intersection* isects,
    global int* groupBuffer
)
{
    int id= get_global_id( 0 );
    
    global Intersection* isect = isects + id;

    if(isect->hit)
    {
        groupBuffer[id] = getMaterial(isects[id].mat);//getMaterial(isects[id].mat);

    }    
    else
        groupBuffer[id] = -1;
}

__kernel void findBound(
    //group to look for instances
    global const int* groupIndex,

     //mesh
    global const float4* points,
    global const Face*   faces,
    global const int*    size,

    //global bound of size 6 -> xmin, ymin, zmin, xmax, ymax, zmax
    global const float* groupBound
)
{
    //global id  for mesh at index (id)
    int id= get_global_id( 0 );

    //Scene mesh
    TriangleMesh mesh = {points, faces, size[0]};

    //Get face at id
    global Face * face = faces + id;
    
    //Get bound coordinates
    global float* xmin = groupBound + 0;
    global float* ymin = groupBound + 1;
    global float* zmin = groupBound + 2;
    global float* xmax = groupBound + 3;
    global float* ymax = groupBound + 4;
    global float* zmax = groupBound + 5;


    //update bounds
    int groupID = getMaterial(face-> mat);
    //printlnInt(groupID);
    if(groupIndex[0] == groupID)
    {
        //printlnInt(*groupIndex);
        BoundingBox bounds = getBoundingBox(mesh, id);

        //bound->minimum = min(bound->minimum, point);
        atomicMin(xmin, bounds.minimum.x);
        atomicMin(ymin, bounds.minimum.y);
        atomicMin(zmin, bounds.minimum.z);

        //bound->maximum = max(bound->maximum, point);
        atomicMax(xmax, bounds.maximum.x);
        atomicMax(ymax, bounds.maximum.y);
        atomicMax(zmax, bounds.maximum.z);
        
        BoundingBox box;
        box.minimum = (float4)(*xmin, *ymin, *zmin, 0);
        box.maximum = (float4)(*xmax, *ymax, *zmax, 0);
        //printBound(box);
    }
}
