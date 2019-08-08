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

    //pixel value
    float2 pixel = getPixel(id, width[0], height[0]);

    //camera matrix, m = world_to_view, mInv = view_to_world
    transform camera_matrix = camera_transform(camera->position, camera->lookat, camera->up);

    //get global ray
    global Ray* r = rays + id;

    //distance to ndc and then aspect ratio
    float d = 1.0f/tan(radians((*camera).fov)/2.0f);
    float a = 10; //width[0]/height[0];

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
    
    if(id < *count)
    {
      if(isRayActive(*ray))
      {
        //intersect
        bool hit = intersectGlobal(ray, isect, mesh, nodes, bounds);    
  
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

__kernel void lightHit(
    global Intersection* isects,
    global Material* materials,
    global float4* accum,
    global int* width,
    global int* height
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //get intersection and material
    global Intersection* isect = isects + id;
    global Material* material = materials + isect->mat;
    
    //if there was an intersection of light
    if(isect->hit && material->emitterEnabled)
    {
        int index = isect->pixel.x + width[0] * isect->pixel.y;
        accum[index] += sampledMaterialColor(*material);
    }
}

// select brdf from material
__kernel void sampleBRDF(
    global Intersection* isects,
    global Material* materials,
    global int* count      //ray count
)
{
    //get thread id
    int id = get_global_id( 0 );  

    if(id < *count)
    {
        //get intersection and material
        global Intersection* isect = isects + id;
        global Material* material = materials + isect->mat;
        
        isect->sampled_brdf = selectBRDF(*material);
    }
}

__kernel void updateFrameImage(
    global float4* accum,
    global int* frame,
    global float* frameCount
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //update frame render
    frame[id] = getIntARGB((float4)(accum[id].xyz/frameCount[0], 1.f));
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

__kernel void shadeBackground(
    global Intersection* isects,
    global int* width,
    global int* height,
    global int* imageBuffer
)
{
    //get thread id
    int id = get_global_id( 0 );

    //updated the intersected areas color
    global Intersection* isect = isects + id;
    if(!isect->hit)
    {
        //pixel index
        int index = isect->pixel.x + width[0] * isect->pixel.y;
        //update
        imageBuffer[id] = getIntARGB((float4)(0, 0, 0, 1));
    }
}

__kernel void updateShadeImage(
    global Intersection* isects,
    global int* width,
    global int* height,
    global int* imageBuffer
)
{
    int id= get_global_id( 0 );

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

__kernel void updateNormalShadeImage(
    global Intersection* isects,
    global int* width,
    global int* height,
    global int* imageBuffer
)
{
    int id= get_global_id( 0 );

    //updated the intersected areas color
    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        //shade normal if facing camera or not
        float ndotd = dot(isect->d, isect->n);
        float4 shade = ndotd < 0 ? (float4)(1, 0, 0, 1) : (float4)(0, 0, 1, 1);
        shade.xyz *= fabs(ndotd);

        //pixel index
        int index = isect->pixel.x + width[0] * isect->pixel.y;

        //update
       imageBuffer[index] = getIntARGB(shade);
    }
}


__kernel void groupBufferPass(
    global Intersection* isects,
    global int* width,
    global int* height,
    global int* groupBuffer
)
{
    int id= get_global_id( 0 );

    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        int index = isect->pixel.x + width[0] * isect->pixel.y;
        groupBuffer[index] = getMaterial(isect->mat);

    }    

}

