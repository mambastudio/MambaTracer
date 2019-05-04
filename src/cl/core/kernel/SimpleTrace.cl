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

__kernel void debug(
    global CameraStruct* camera,
    global float2* pixels)
{

}

/*
    - Essential Mathematics for Games and Interactive Applications: 2nd edition, pg 203
    - Soon to expand for thin lens, which is trivial

    - Notice we do many operations on global memory level to avoid private memory cache strain
      or it will render glitches as it came to my understanding
*/
__kernel void generateCameraRays(
    global CameraStruct* camera,
    global Ray* rays,
    global int* width,
    global int* height)
{
    //global id and pixel making
    int id= get_global_id( 0 );
    float2 pixel = getPixel(id, width[0], height[0]);
    
    //camera matrix, m = world_to_view, mInv = view_to_world
    transform camera_matrix = camera_transform(camera->position, camera->lookat, camera->up);

    //get global ray
    global Ray* r = rays + id;

    //distance to ndc and then aspect ratio
    float d = 1.0f/tan(radians((*camera).fov)/2.0f);
    float a = width[0]/height[0];

    //direction (px, py, pz, 0) and origin (0, 0, 0, 0)
    r->d = normalize((float4)(a * (2.f * pixel.x/width[0] - 1), -2.f * pixel.y/height[0] + 1, -d, 0));
    r->o = 0;  //will be important for global illumination, when we reuse the rays

    //transform to world space
    r->o = transform_point4(camera_matrix.mInv, r->o);
    r->d = transform_vector4(camera_matrix.mInv, r->d);

    //init ray
    initGlobalRay(r, r->o, r->d);

    //set pixel & active
    r->pixel = pixel;
    r->extra.x = true;
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

    //get ray, create both isect and mesh
    global Ray* ray = rays + id;
    global Intersection* isect = isects + id;
    TriangleMesh mesh = {points, normals, faces, size[0]};
    
    if(isRayActive(*ray))
    {
      //intersect
      bool hit = intersectGlobal(ray, isect, mesh, nodes, bounds);    
      
      //update hit status and what pixel it represent
      isect->pixel = ray->pixel;
      isect->hit = hit;
    }
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
        color.xyz   = getMaterialColor(materials[isect->mat], coeff).xyz;
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

    //first set the image buffer to black
    imageBuffer[id] = getIntARGB((float4)(0, 0, 0, 1));
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    //updated the intersected areas color
    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        //pixel index
        int index = isect->pixel.x + width[0] * isect->pixel.y;
        //update
       imageBuffer[index] = getIntARGB(isect->throughput);
    }

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
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,

    //global bound of size 6 -> xmin, ymin, zmin, xmax, ymax, zmax
    global const float* groupBound
)
{
    //global id  for mesh at index (id)
    int id= get_global_id( 0 );

    //Scene mesh
    TriangleMesh mesh = {points, normals, faces, size[0]};

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
