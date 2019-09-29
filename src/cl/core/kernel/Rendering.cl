__kernel void testRandom(
    global int*   frameBuffer,
    global int*   seed
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //seed for this thread
    int seedThread  = WangHash(*seed * id);

    //get pixel
    int2 pixel = random_int2_range(&seedThread, 800, 600);
    
    //set pixel color
    frameBuffer[pixel.x + pixel.y * 800] = getIntARGB((float4)(1, 0, 0, 1));

}

//mark the intersects and update path bsdf
__kernel void UpdateBSDFIntersect(
    global Intersection* isects,
    global Ray* rays,
    global Path* paths,
    global int* width,
    global int* height,
    global int* num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
        global Intersection* isect = isects + global_id;
        int index                  = isect->pixel.x + width[0] * isect->pixel.y;
        global Ray* ray            = rays + index;
        global Path* path          = paths + index;

        if(isect->hit && path->active)
        {
            path->bsdf   = setupBSDF(ray, isect);
            isect->sampled_brdf = 1;
        }
        else
        {
            path->active = false;
            isect->sampled_brdf = 0;
        }
    }
}

__kernel void LightHitPass(
    global Intersection* isects,
    global Path*         paths,
    global Material*     materials,
    global float4*       accum,
    global int*          width,
    global int*          height,
    global int*          num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
       global Intersection* isect = isects + global_id;

       if(isect->hit)
       {
           int pixelIndex            = isect->pixel.x + width[0] * isect->pixel.y;
           int pathIndex             = isect->pixel.x + width[0] * isect->pixel.y;
           
           global Material* material = materials + isect->mat;

           if(isEmitter(*material))
           {
               global Path* path         = paths + pathIndex;
               
               if(path->active)
               {
                  float4 contribution = path->throughput * getEmitterColor(*material);
                  atomicAddFloat4(&accum[pixelIndex], contribution);    //add to accumulator
               }

               //we are done with this intersect and path
               isect->sampled_brdf = 0;
               path->active = false;
           }
       }
    }
}

__kernel void EvaluateBSDFIntersect(
    global Intersection* isects,
    global Path* paths,
    global Material* materials,
    global int* width,
    global int* height,
    global int* num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
        global Intersection* isect = isects + global_id;                int index = isect->pixel.x + width[0] * isect->pixel.y;
        global Path* path          = paths + index;
        global Material* material  = materials + isect->mat;

        atomicMulFloat4(&path->throughput, sampledMaterialColor(*material));  //mul
        


    }

}
__kernel void SampleBSDFRayDirection(
    global Intersection* isects,
    global Ray*          rays,
    global Path*         paths,
    global int*          width,
    global int*          height,
    global int*          num_rays,
    global int*          seed
)
{
    int global_id = get_global_id(0);

    //seed for this thread
    int seedThread               = WangHash(*seed * global_id);

    //get intersection and index
    global Intersection* isect   = isects + global_id;
    int index = isect->pixel.x + width[0] * isect->pixel.y;

    if(global_id < *num_rays)
    {   
        //path and ray
        global Path* path            = paths + index;
        global Ray* ray              = rays + global_id;

        //random sample direction
        float2 sample                = random_float2(&seedThread);
        float4 d                     = world_coordinate(path->bsdf.frame, sample_hemisphere(sample));
        float4 o                     = isect->p;

        //new ray direction
        initGlobalRay(ray, o, d);
        ray->pixel                   = isect->pixel;
        
        //init isect
        InitIsect(isect);

    }

}


__kernel void UpdateFrameImage(
    global float4* accum,
    global int* frame,
    global float* frameCount
)
{
    //get thread id
    int id = get_global_id( 0 );
    
   // printFloat(*frameCount);

    //update frame render
    frame[id] = getIntARGB((float4)(accum[id].xyz/frameCount[0], 1.f));
}