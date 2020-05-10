void addAccum(__global float4* accum, float4 value)
{
    if(isFloat4AbsValid(value))
        atomicAddFloat4(accum, value);
}

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

__kernel void LightHitPass(global Intersection* isects,
                           global Ray*          rays,
                           global Path*         paths,
                           global int*          totalLights,

                           //mesh
                           global Material*     materials,
                           global const float4* points,
                           global const float4* normals,
                           global const Face*   faces,
                           global const int*    size,

                           global float4*       accum,
                           global int*          pixel_indices,
                           global int*          num_isects)
{
    int global_id = get_global_id(0);
    TriangleMesh mesh    = {points, normals, faces, size[0]};

    if(global_id < *num_isects)
    {
       global Intersection* isect = isects + global_id;
       global Ray* ray            = rays + global_id;
       global int* index          = pixel_indices + global_id;
       global Path* path          = paths + *index;

       if(isect->hit)
       {
           //UPDATE ALL BSDF FIRST IN PATH (could have been done in another kernel but that's waste of code)
           path->bsdf                = setupBSDF(ray, isect);

           //deal with emitter
           global Material* material = materials + isect->mat;
           if(isEmitter(*material))
           {
              //light pick probability
              float lightPickProb = 1.f / *totalLights;
              //get area light and contribution
              AreaLight aLight = getAreaLight(materials, mesh, isect->id);
              float directPdfA;
              float4 contrib = getRadianceAreaLight(aLight, ray->d, isect->p, &directPdfA);
              if(isFloat3Zero(contrib.xyz))
                  return;
              
              //weight path
              float misWeight = 1.f;
              if(!path->lastSpecular)
              {
                   float directPdfW = pdfAtoW(directPdfA, ray->tMax, path->bsdf.localDirFix.z);
                   misWeight = mis2(path->lastPdfW, directPdfW * lightPickProb);
              }
              //accum[*index]       += contribution;
              addAccum(&accum[*index], path->throughput * contrib * misWeight);
              //we are done with this intersect and path
              isect->hit = false;
           }
       }
    }
}

__kernel void SampleBSDFRayDirection(global Intersection* isects,
                                     global Ray*          rays,
                                     global Path*         paths,  
                                     global Material*     materials,
                                     global int*          pixel_indices,
                                     global State*        state,
                                     global int*          num_rays)
{
    int global_id = get_global_id(0);

    //seeds for each thread
    int2 seed = generate_seed(state);
    //get intersection and path_index
    global Intersection* isect   = isects + global_id;
    global int* path_index       = pixel_indices + global_id;
    global Path* path            = paths + *path_index;
    global Material* material    = materials + path->bsdf.materialID;
    BSDF bsdf                    = path->bsdf;

    if(global_id < *num_rays)
    {     
        //path and ray
        global Ray* ray              = rays + global_id;

        //random sample direction
        float2 sample                = random_float2(&seed);

        //bsdf  sampling
        float4 dir;
        float pdf, cosThetaOut;
        float4 factor               = sampleBrdf(*material, bsdf, sample, &dir, &cosThetaOut, &pdf);

        //path contribution (light equation)
        path->throughput             *= factor * (cosThetaOut / pdf);

        //init new ray direction
        float4 d                     = dir;
        float4 o                     = isect->p;
        
        //set specular false for now
        path->lastSpecular           = false;
        path->lastPdfW               = pdf;

        //new ray direction
        initGlobalRay(ray, o, d);
        InitIsect(isect);
    }
}

__kernel void DirectLight(
    global Path*         paths,
    global Intersection* isects,
    global Light*        lights,
    global int*          totalLights,
    global Ray*          occlusRays,

    //accumulation
    global float4*       accum,
    global int*          pixel_indices,
    
    //count
    global int*          activeCount,
    global State*        state,

    //mesh and material
    global Material*     materials,
    global const float4* points,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,
    
    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds,
    
    //start node
    global const int* startNode
)
{

    int global_id        = get_global_id(0);
    TriangleMesh mesh    = {points, normals, faces, size[0]};
    
    //get intersection and path_index
    global Intersection* isect   = isects + global_id;
    global Ray* ray              = occlusRays + global_id;
    global int* pixel_index       = pixel_indices + global_id;
    global Path* path            = paths + *pixel_index;

    if(global_id < *activeCount)
    {        
        //seeds for each thread
        int2 seed = generate_seed(state);

        //for light surface sample
        float2 sample                = random_float2(&seed);
         //sample light index uniformly
        int lightIndex = random_int_range(&seed, *totalLights);
        //light and index of mesh
        global Light* light = lights + lightIndex;
        int triangleIndex   = light->faceId;
        //light pick probability
        float lightPickProb = 1.f / *totalLights;
        //get area light
        AreaLight aLight = getAreaLight(materials, mesh, triangleIndex);
        //material
        global Material* material    = materials + path->bsdf.materialID;
        //radiance from direct light
        float4 directionToLight = makeFloat4(0, 0, 0, 0);
        float distance, directPdfW;
        float4 radiance = illuminateAreaLight(aLight, isect->p, sample, &directionToLight, &distance, &directPdfW);

        if(!isFloat3Zero(radiance.xyz))
        {
            float bsdfPdfW, cosThetaOut;
            float4 factor =  evaluateBrdf(*material, path->bsdf, directionToLight, &cosThetaOut, &bsdfPdfW);

            if(!isFloat3Zero(factor.xyz))
            {
                float4 contrib = makeFloat4(0, 0, 0, 0);    //important since undeclared variable might have issues when used
                float weight = 1.f;
                weight = mis2(directPdfW * lightPickProb, bsdfPdfW);

                contrib.xyz    = (weight * cosThetaOut / (lightPickProb * directPdfW)) *
                                 (radiance.xyz * factor.xyz) * path->throughput.xyz;
                //new ray direction
                initGlobalRayT(ray, isect->p, directionToLight, distance - 2*EPS_RAY);

                //test occlusion
                if(!testOcclusion(ray, mesh, nodes, bounds, startNode))
                {
                    addAccum(&accum[*pixel_index], contrib);
                }
            }
        }
    }
}