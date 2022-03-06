void addAccum(__global float4* accum, float4 value)
{

    if(isFloat4AbsValid(value))
        atomicAddFloat4(accum, value);

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
              Bsdf bsdf            = setupBsdf(ray, isect, materials);
              
              if(isBsdfInvalid(bsdf))
              {
                  //we are done with this intersect and path
                  isect->hit = false;
                  return;
              }
              path->bsdf           = bsdf;

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
                           global LightGrid*        lightGrid)
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
           Bsdf bsdf = path->bsdf;
           if(isBsdfEmitter(bsdf))
           {
              //light pick probability
              float lightPickProb = 1.f / *totalLights;
              //get area light and contribution
              AreaLight aLight = getAreaLight(bsdf, mesh, isect->id);
              float directPdfA = 0;
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
       else if(lightGrid->isPresent)
       {
            EnvironmentLight aLight = getEnvironmentLight(lightGrid, envmap,  lum, lumsat);
            float directPdfW;
                     
            float4 contrib = getRadianceEnvironmentLight(aLight, ray->d, path->hitpoint, &directPdfW);
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

    Bsdf bsdf                  = path->bsdf;

    tex->parameters.x = bsdf.materialID;               //to find image in CPU during texture lookup search in host code

    if(id < *count)
    {
        //diffuse
        if(bsdf.param.isDiffuseTexture)
        {
              tex->diffuseTexture.w = true;
              tex->diffuseTexture.x = castFloatToInt(isect->uv.x);
              tex->diffuseTexture.y = castFloatToInt(isect->uv.y);
        }
        else
        {
              tex->diffuseTexture.w = false;   //no texture
        }
        
        //glossy
        if(bsdf.param.isGlossyTexture)
        {
              tex->glossyTexture.w = true;
              tex->glossyTexture.x = castFloatToInt(isect->uv.x);
              tex->glossyTexture.y = castFloatToInt(isect->uv.y);
        }
        else
        {
              tex->glossyTexture.w = false;   //no texture
        }
        
        //roughness
        if(bsdf.param.isRoughnessTexture)
        {
    
              tex->roughnessTexture.w = true;
              tex->roughnessTexture.x = castFloatToInt(isect->uv.x);
              tex->roughnessTexture.y = castFloatToInt(isect->uv.y);
        }
        else
        {
              tex->roughnessTexture.w = false;   //no texture
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

    if(id < *count)
    {
        //diffuse
        if(tex->diffuseTexture.w)
        {
           float4 texColor = getFloat4ARGB(tex->diffuseTexture.z);
           path->bsdf.param.diffuse_color     = texColor;
        }
        
        //glossy
        if(tex->glossyTexture.w)
        {
           float4 texColor = getFloat4ARGB(tex->glossyTexture.z);
           path->bsdf.param.glossy_color     = texColor;
        }
        
        //roughness
        if(tex->roughnessTexture.w)
        {
           float4 texColor = getFloat4ARGB(tex->roughnessTexture.z);
           path->bsdf.param.glossy_param.y  = max(0.001f, texColor.x);
           path->bsdf.param.glossy_param.z  = max(0.001f, texColor.y);
         //  printFloat(roughness);
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
    Bsdf bsdf                    = path->bsdf;

    if(global_id < *num_rays)
    {     
        //path and ray
        global Ray* ray              = rays + global_id;

        //random sample direction
        float3 sample                = random_float3(&seed);

        //bsdf  sampling
        float4 dir;
        float pdf, cosThetaOut;
        float4 factor               = SampleBsdf(bsdf, sample, &dir, &pdf, &cosThetaOut);
        


        //path contribution (light equation)
        path->throughput             *= factor * (cosThetaOut / pdf);

        //init new ray direction
        float4 d                     = dir;
        float4 o                     = isect->p;
        
        //set specular false for now
        path->lastSpecular           = false;
        path->lastPdfW               = pdf;

        //last hit point (good for calculating the light grid cell)
        path->hitpoint               = isect->p;

        //new ray direction
        initGlobalRay(ray, o, d);
        InitIsect(isect);
    }
}

float4 illuminateMeshLight(global Intersection* isect,
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
     Bsdf lightBSDF = setupBSDFAreaLight(materials, mesh, triangleIndex);
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
                        float*               directPdfW,
                        float                *luminance,
                        int                  *lightGridIndex)
{
    *lightPickProb    =         1.f;
    //for light surface sample
    float4 sample                = random_float4(seed);

    //sample either environment light or mesh light
    if(eLight.lightGrid->isPresent && (*totalLights > 0))
    {
         float rnd         = get_random(seed);
         bool sampleALight = (rnd < 0.5f);
         
         //light mesh or infinite light
         *lightPickProb    *= 0.5f;

         if(sampleALight)
         {
              float4 radiance = illuminateMeshLight(isect, mesh, materials, lightInfo, totalLights,
                                                       sample.xy, lightPickProb, seed, directionToLight, distance, directPdfW);
              return radiance;
         }
         else
         {
              float4 radiance = illuminateEnvironmentLight(eLight, isect->p, sample, directionToLight, distance, directPdfW, luminance, lightGridIndex);
              return radiance;
         }
    }
    //sample environment map if present
    if(eLight.lightGrid->isPresent)
    {
        float4 radiance = illuminateEnvironmentLight(eLight, isect->p, sample, directionToLight, distance, directPdfW, luminance, lightGridIndex);
        return radiance;
    }
    //sample mesh light if present
    else if(*totalLights > 0)
    {
        //radiance from direct light
        float4 radiance = illuminateMeshLight(isect, mesh, materials, lightInfo, totalLights,
                                                 sample.xy, lightPickProb, seed, directionToLight, distance, directPdfW);
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
    global LightGrid*        lightGrid
)
{

    int global_id            = get_global_id(0);
    TriangleMesh mesh        = {points, uvs, normals, faces, size[0]};
    EnvironmentLight eLight  = getEnvironmentLight(lightGrid, envmap,  lum, lumsat);
    
    //get intersection and path_index
    global Intersection* isect   = isects + global_id;
    global Ray* ray              = occlusRays + global_id;
    global int* pixel_index      = pixel_indices + global_id;
    global Path* path            = paths + *pixel_index;
    Bsdf bsdf                    = path->bsdf;
    
    float            luminance;
    int              lightGridIndex;


    if(global_id < *activeCount)
    {        
        //seeds for each thread
        int2 seed = generate_seed(state);
        
        float4 directionToLight = (float4)(0, 0, 0, 0);
        float distance, directPdfW, lightPickProb;

        //get radiance from direct light sampling
        float4 radiance = illuminateLight(isect, eLight, mesh, materials, lights, totalLights,
                                                 &lightPickProb, &seed, &directionToLight, &distance, &directPdfW, &luminance, &lightGridIndex);

        if(!isFloat3Zero(radiance.xyz))
        {
            float bsdfPdfW = 0, cosThetaOut = 0;
            float4 factor =  EvaluateBsdf(path->bsdf, directionToLight, &bsdfPdfW, &cosThetaOut);

            if(!isFloat3Zero(factor.xyz))
            {
                float4 contrib = makeFloat4(0, 0, 0, 1);    //important since undeclared variable might have issues when used
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
                    //accumLightGrid(eLight);
                    //accumLightGrid(eLight, luminance, lightGridIndex);
                    addAccum(&accum[*pixel_index], contrib);
                }
            }
        }

    }
}





