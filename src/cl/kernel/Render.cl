typedef struct
{
    float4 throughput;        //throughput (multiplied by emission)
    float4 hitpoint;             //position of vertex
    int    pathlength;           //path length or number of segment between source and vertex
    int    lastSpecular;
    float  lastPdfW;
    int    active;               //is path active
    BSDF   bsdf;                 //bsdf (stores local information together with incoming direction)

}Path;

void addAccum(__global float4* accum, float4 value)
{

    if(isFloat4AbsValid(value))
        atomicAddFloat4(accum, value);

}

__kernel void InitPathData(
    global Path* paths
)
{
    //get global id
    int global_id = get_global_id(0);
    //get path
    global Path* path = paths + global_id;
    //initialize path
    path->throughput          = makeFloat4(1, 1, 1, 1);
    path->lastSpecular        = true;
    path->lastPdfW            = 1;
    path->active              = true;
    path->bsdf.materialID     = -1;
    path->bsdf.frame.mX       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mY       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mZ       = makeFloat4(0, 0, 0, 0);
    path->bsdf.localDirFix    = makeFloat4(0, 0, 0, 0);
}

// this is where you select the required bsdf, portal for filtering later
__kernel void SetupBSDFPath(global Intersection* isects,
                        global Ray*          rays,
                        global Path*         paths,
                        global Material*     materials,
                        global int*          pixel_indices,      //used coz of compaction
                        global int*          num_isects)
{
    int global_id = get_global_id(0);
    
    if(global_id < *num_isects)
    {
        global Intersection* isect = isects + global_id;
        global Ray* ray            = rays + global_id;
        global int* index          = pixel_indices + global_id;
        global Path* path          = paths + *index;
      
        if(isect->hit)
        {
              path->bsdf           = setupBSDF(ray, isect, materials);
        }
    }
}

__kernel void LightHitPass(global Intersection*     isects,
                           global Ray*              rays,
                           global Path*             paths,
                           global int*              totalLights,

                           //mesh
                           global Material*         materials,
                           global const float4*     points,
                           global const float2*     uvs,
                           global const float4*     normals,
                           global const Face*       faces,
                           global const int*        size,

                           global float4*           accum,
                           global int*              pixel_indices,
                           global int*              num_isects,

                           global float4*           envmap,
                           global float*            lum,
                           global float*            lumsat,
                           global EnvironmentGrid*  envgrid)
{
    int global_id = get_global_id(0);
    TriangleMesh mesh    = {points, uvs, normals, faces, size[0]};

    if(global_id < *num_isects)
    {
       global Intersection* isect = isects + global_id;
       global Ray* ray            = rays + global_id;
       global int* index          = pixel_indices + global_id;
       global Path* path          = paths + *index;

       if(isect->hit)
       {
           //deal with emitter
           BSDF bsdf = path->bsdf;
           if(bsdf.param.brdfType == EMITTER)
           {
              //light pick probability
              float lightPickProb = 1.f / *totalLights;
              //get area light and contribution
              AreaLight aLight = getAreaLight(bsdf, mesh, isect->id);
              float directPdfA;
              float4 contrib = getRadianceAreaLight(aLight, ray->d, isect->p, &directPdfA);
              if(isFloat3Zero(contrib.xyz))
                  return;
              
              //weight path
              float misWeight = 1.f;
              if(!path->lastSpecular)
              {
                   float directPdfW = pdfAtoW(directPdfA, ray->tMax, bsdf.localDirFix.z);
                   misWeight = mis2(path->lastPdfW, directPdfW * lightPickProb);
              }
             
              //accumulate radiance to screenspace buffer
              addAccum(&accum[*index], path->throughput * contrib * misWeight);
              //we are done with this intersect and path
              isect->hit = false;
           }
       }
       else if(envgrid->isPresent)
       {
            EnvironmentLight aLight = getEnvironmentLight(envgrid, envmap,  lum, lumsat);
            float directPdfW;
            
            float4 contrib = getRadianceEnvironmentLight(aLight, ray->d, isect->p, &directPdfW);
            if(isFloat3Zero(contrib.xyz))
                 return;
        
            //weight path
            float misWeight = 1.f;
            if(!path->lastSpecular)
            {
                 misWeight = mis2(path->lastPdfW, directPdfW);
            }
            //accumulate radiance to screenspace buffer
            addAccum(&accum[*index], path->throughput * contrib * misWeight);
       }
    }
}

__kernel void texturePassGI(
    global Path*         paths,
    global Intersection* isects,
    global TextureData*  texData,
    
    global int*          pixel_indices,
    global int*          count
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //get intersection and material
    global Intersection* isect = isects + id;
    global int* index          = pixel_indices + id;
    global TextureData* tex    = texData + id;
    global Path* path          = paths + *index;

    BSDF bsdf                  = path->bsdf;


    if(id < *count)
    {
        if(bsdf.param.isTexture)
        {
              tex->hasBaseTex        = true;
              tex->materialID        = bsdf.materialID;            //to find image in CPU during texture lookup search in host code
    
              tex->baseTexture.x     = castFloatToInt(isect->uv.x);
              tex->baseTexture.y     = castFloatToInt(isect->uv.y);
            //tex->baseTexture.z is argb
        }
        else
        {
              tex->hasBaseTex        = false;
              tex->hasOpacity        = false;
        }
    }
}

__kernel void updateToTextureColorGI(
    global Path*         paths,
    global TextureData*  texData,

    global int*          pixel_indices,
    global int*          count
)
{
    //get thread id
    int id = get_global_id( 0 );

    //texture for pixel
    global int* index          = pixel_indices + id;
    global TextureData* tex    = texData + id;
    global Path* path          = paths + *index;

    float4 texColor = getFloat4ARGB(tex->baseTexture.z);
    
    if(id < *count)
        if(tex->hasBaseTex)
             path->bsdf.param.base_color     = texColor;
    

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
        float4 factor               = sampleBrdf(bsdf, sample, &dir, &cosThetaOut, &pdf);

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

float4 directAreaLight(global Intersection* isect,
                       //mesh light
                       TriangleMesh         mesh,
                       global Material*     materials,
                       global LightInfo*    lightInfo,
                       global int*          totalLights,
                       //other parameters for transport calculation
                       float2               sample,
                       float*               lightPickProb,
                       int2*                seed,
                       float4*              directionToLight,
                       float*               distance,
                       float*               directPdfW)
{
     //sample light index uniformly
     int lightIndex = random_int_range(seed, *totalLights);
     //light and index of mesh
     global LightInfo* light = lightInfo + lightIndex;
     int triangleIndex   = lightInfo->faceId;
     //light pick probability
     *lightPickProb *= 1.f / *totalLights;
     //light bsdf
     BSDF lightBSDF = setupBSDFAreaLight(materials, mesh, triangleIndex);
     //get area light
     AreaLight aLight = getAreaLight(lightBSDF, mesh, triangleIndex);
     //radiance from direct light
     float4 radiance = illuminateAreaLight(aLight, isect->p, sample, directionToLight, distance, directPdfW);
     return radiance;
  }

float4 illuminateLight( global Intersection* isect,
                        //environment
                        EnvironmentLight     eLight,
                        //mesh light
                        TriangleMesh         mesh,
                        global Material*     materials,
                        global LightInfo*    lightInfo,
                        global int*          totalLights,
                        //other parameters for transport calculation
                        float*               lightPickProb,
                        int2*                seed,
                        float4*              directionToLight,
                        float*               distance,
                        float*               directPdfW)
{
    *lightPickProb    =         1.f;
    //for light surface sample
    float2 sample                = random_float2(seed);

    //sample either environment light or mesh light
    if(eLight.envgrid->isPresent && (*totalLights > 0))
    {
         float rnd         = get_random(seed);
         bool sampleALight = (rnd < 0.5f);
         
         //light mesh or infinite light
         *lightPickProb    *= 0.5f;

         if(sampleALight)
         {
              float4 radiance = directAreaLight(isect, mesh, materials, lightInfo, totalLights, 
                                                       sample, lightPickProb, seed, directionToLight, distance, directPdfW);
              return radiance;
         }
         else
         {
              float4 radiance = illuminateEnvironmentLight(eLight, isect->p, sample, directionToLight, distance, directPdfW);
              return radiance;
         }
    }
    //sample environment map if present
    if(eLight.envgrid->isPresent)
    {
        float4 radiance = illuminateEnvironmentLight(eLight, isect->p, sample, directionToLight, distance, directPdfW);
        return radiance;
    }
    //sample mesh light if present
    else if(*totalLights > 0)
    {
        //radiance from direct light
        float4 radiance = directAreaLight(isect, mesh, materials, lightInfo, totalLights,
                                                 sample, lightPickProb, seed, directionToLight, distance, directPdfW);
        return radiance;
    }
}

__kernel void DirectLight(
    global Path*         paths,
    global Intersection* isects,
    global LightInfo*    lights,  //for sampling deterministically for shadow test
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
    global const float2* uvs,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,
    
    //bvh
    global const BVHNode* nodes,
    global const BoundingBox* bounds,
    
    global float4*           envmap,
    global float*            lum,
    global float*            lumsat,
    global EnvironmentGrid*  envgrid
)
{

    int global_id            = get_global_id(0);
    TriangleMesh mesh        = {points, uvs, normals, faces, size[0]};
    EnvironmentLight eLight  = getEnvironmentLight(envgrid, envmap,  lum, lumsat);
    
    //get intersection and path_index
    global Intersection* isect   = isects + global_id;
    global Ray* ray              = occlusRays + global_id;
    global int* pixel_index      = pixel_indices + global_id;
    global Path* path            = paths + *pixel_index;
    BSDF bsdf                    = path->bsdf;

    if(global_id < *activeCount)
    {        
        //seeds for each thread
        int2 seed = generate_seed(state);
        
        float4 directionToLight = (float4)(0, 0, 0, 0);
        float distance, directPdfW, lightPickProb;

        //get radiance from direct light sampling
        float4 radiance = illuminateLight(isect, eLight, mesh, materials, lights, totalLights,
                                                 &lightPickProb, &seed, &directionToLight, &distance, &directPdfW);

        if(!isFloat3Zero(radiance.xyz))
        {
            float bsdfPdfW, cosThetaOut;
            float4 factor =  evaluateBrdf(path->bsdf, directionToLight, &cosThetaOut, &bsdfPdfW);

            if(!isFloat3Zero(factor.xyz))
            {
                float4 contrib = makeFloat4(0, 0, 0, 0);    //important since undeclared variable might have issues when used
                float weight = 1.f;
                //weight = mis2(directPdfW * lightPickProb, bsdfPdfW);
                weight = mis2(directPdfW, bsdfPdfW);

                contrib.xyz    = (weight * cosThetaOut / (lightPickProb * directPdfW)) *
                                 (radiance.xyz * factor.xyz)* path->throughput.xyz;
                //new ray direction
                initGlobalRayT(ray, isect->p, directionToLight, distance - 2*EPS_RAY);

                //test occlusion
                if(!testOcclusion(ray, mesh, nodes, bounds))
                {
                    addAccum(&accum[*pixel_index], contrib);
                }
            }
        }

    }
}





