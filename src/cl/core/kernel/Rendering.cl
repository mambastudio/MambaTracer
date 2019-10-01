/*
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
*/

float4 ACESFilm(float4 x)
{
    float4 toned;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    toned = clamp((float4)((x.xyz*(a*x.xyz+b))/(x.xyz*(c*x.xyz+d)+e), 1), 0.f, 1.f);
    return toned;
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
    global int*          random0,
    global int*          random1,
    global float*        frameCount
)
{
    int global_id = get_global_id(0);
    
    unsigned int x_coord = global_id % width[0];			/* x-coordinate of the pixel */
    unsigned int y_coord = global_id / width[0];			/* y-coordinate of the pixel */


    //seeds for this thread
    unsigned int seed0 = x_coord * (int)(frameCount[0]) % 1000 + (random0[0] * 100);
    unsigned int seed1 = y_coord * (int)(frameCount[0]) % 1000 + (random1[0] * 100);

    //get intersection and index
    global Intersection* isect   = isects + global_id;
    int index = isect->pixel.x + width[0] * isect->pixel.y;

    if(global_id < *num_rays)
    {   
        //path and ray
        global Path* path            = paths + index;
        global Ray* ray              = rays + global_id;

        //random sample direction
        float2 sample                = random_float2(&seed0, &seed1);
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
   float4 toned = ACESFilm((float4)(accum[id].xyz/frameCount[0], 1.f));

    //update frame render
    frame[id] = getIntARGB(toned);
}