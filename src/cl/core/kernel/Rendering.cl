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
float luminance(float4 v)
{
    // Luminance
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

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
    global int* pixel_indices,
    global int* num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
        global Intersection* isect = isects + global_id;
        global int* index          = pixel_indices + global_id;
        global Ray* ray            = rays + *index;
        global Path* path          = paths + *index;

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
    global CameraStruct* camera,
    global int*          num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
       global Intersection* isect = isects + global_id;

       if(isect->hit)
       {
           int pixelIndex            = isect->pixel.x + camera->dimension.x * isect->pixel.y;
           int pathIndex             = isect->pixel.x + camera->dimension.x * isect->pixel.y;
           
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
    global CameraStruct* camera,
    global int* num_isects
)
{
    int global_id = get_global_id(0);

    if(global_id < *num_isects)
    {
        global Intersection* isect = isects + global_id;                int index = isect->pixel.x + camera->dimension.x * isect->pixel.y;
        global Path* path          = paths + index;
        global Material* material  = materials + isect->mat;

        atomicMulFloat4(&path->throughput, sampledMaterialColor(*material));  //mul
    }

}

__kernel void EvaluateBsdfExplicit( 
    global Intersection*  isects,
    global int*           hits,
    global Path*          paths,
    global Material*      materials,
    global int*           pixel_indices,
    global int*           num_isects
)
{
    int global_id     = get_global_id(0);
    global int* hit   = hits + global_id;

    if(global_id < *num_isects)
    {  
        if(*hit)
        {
            global Intersection* isect   = isects + global_id;
            global int* index            = pixel_indices + global_id;
            global Path* path            = paths + *index;
            global Material* material    = materials + isect->mat;
    
            atomicMulFloat4(&path->throughput, (float4)(1.f, 1.f, 1.f, 1.f));  //mul
        }
    }

}

__kernel void sampleLight(
    global Path*         lightPaths,
    global Light*        lights,
    global int*          totalLights,
    
    //count
    global int*          activeCount,
    global State*        state,

    //mesh and material
    global Material*     materials,
    global const float4* points,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size
)
{
    int global_id        = get_global_id(0);
    
    if(global_id < *activeCount)
    {
        TriangleMesh mesh    = {points, normals, faces, size[0]};
        
        //we assume maximum light sampling is equal to image size
        unsigned int x_coord = global_id % (int)state->dimension.x;			/* x-coordinate of the pixel */
        unsigned int y_coord = global_id / (int)state->dimension.x;			/* y-coordinate of the pixel */
    
        //seeds for this thread
        int2 seed;
        seed.x = x_coord * (int)(state->frameCount) % 1000 + (state->seed.x * 100);
        seed.y = y_coord * (int)(state->frameCount) % 1000 + (state->seed.y * 100);
        
        //for light surface sample
        float2 sample                = random_float2(&seed);


        //sample light index uniformly
        int triangleIndex = random_int_range(&seed, *totalLights);
       
        //get triangle points
        float4 p1 = getP1(mesh, triangleIndex);
        float4 p2 = getP2(mesh, triangleIndex);
        float4 p3 = getP3(mesh, triangleIndex);
    
        //light point
        float4 lightpoint = sample_triangle(sample, p1, p2, p3);
        
        //set light data
        global Path *lightPath = lightPaths + global_id;
        lightPath->hitpoint = lightpoint;
    }
}

__kernel void GenerateShadowRays(
    global Path*         lightPaths,
    global Intersection* isects,
    global Ray*          rays,
    global int*          count
)
{
    int global_id = get_global_id(0);
    if(global_id < *count)
    {
        //get global data
        global Path *lightPath       = lightPaths + global_id;
        global Intersection* isect   = isects + global_id;
        global Ray* ray              = rays + global_id;

        //ray data
        float4 d                     = normalize(lightPath->hitpoint - isect->p);
        float4 o                     = isect->p;
        
        //new ray direction
        initGlobalRay(ray, o, d);
    }
}

__kernel void SampleBSDFRayDirection(
    global Intersection* isects,
    global Ray*          rays,
    global Path*         paths,
    global int*          pixel_indices,
    global State*        state,
    global int*          num_rays
)
{
    int global_id = get_global_id(0);

    unsigned int x_coord = global_id % (int)state->dimension.x;			/* x-coordinate of the pixel */
    unsigned int y_coord = global_id / (int)state->dimension.x;			/* y-coordinate of the pixel */


    //seeds for this thread
    int2 seed;
    seed.x = x_coord * (int)(state->frameCount) % 1000 + (state->seed.x * 100);
    seed.y = y_coord * (int)(state->frameCount) % 1000 + (state->seed.y * 100);

    if(global_id < *num_rays)
    {
        //get intersection and path_index
        global Intersection* isect   = isects + global_id;
        global int* path_index       = pixel_indices + global_id;

        //path and ray
        global Path* path            = paths + *path_index;
        global Ray* ray              = rays + global_id;

        //random sample direction
        float2 sample                = random_float2(&seed);
        float4 d                     = world_coordinate(path->bsdf.frame, sample_hemisphere(sample));
        float4 o                     = isect->p;

        //new ray direction
        initGlobalRay(ray, o, d);
        ray->pixel                   = isect->pixel;
        
        //init isect
        InitIsect(isect);

    }

}