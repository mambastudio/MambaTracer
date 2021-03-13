__kernel void InitCameraRayData(
    global CameraStruct* camera,
    global Ray* rays)
{
    //global id and pixel making
    int id= get_global_id( 0 );      

    //pixel value
    float2 pixel = getPixel(id, camera->dimension.x, camera->dimension.y);

    //camera matrix, m = world_to_view, mInv = view_to_world
    transform camera_matrix = camera_transform(camera->position, camera->lookat, camera->up);

    //get global ray
    global Ray* r = rays + id;

    //distance to ndc and then aspect ratio
    float d = 1.0f/tan(radians((*camera).fov)/2.0f);
    float a = camera->dimension.x/camera->dimension.y;

    //direction (px, py, pz, 0) and origin (0, 0, 0, 0)
    r->d = normalize((float4)(a * (2.f * pixel.x/camera->dimension.x - 1), -2.f * pixel.y/camera->dimension.y + 1, -d, 0));
    r->o = 0;  //will be important for global illumination, when we reuse the rays

    //transform to world space
    r->o = transform_point4(camera_matrix.mInv, r->o);
    r->d = transform_vector4(camera_matrix.mInv, r->d);

    //init ray
    initGlobalRay(r, r->o, r->d);

}

// this is where you select the required bsdf, portal for filtering later
__kernel void SetupBSDFRaytrace(global Intersection* isects,
                                global Ray*          rays,
                                global BSDF*         bsdfs,
                                global Material*     materials)
{
    int global_id = get_global_id(0);

    global Intersection* isect = isects + global_id;
    global Ray* ray            = rays + global_id;
    global BSDF* bsdf          = bsdfs + global_id;
  
    if(isect->hit)
    {
          *bsdf                = setupBSDF(ray, isect, materials);
    }
}

__kernel void textureInitPassRT(
    global BSDF*         bsdfs,
    global Intersection* isects,
    global TextureData*  texData
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //get intersection and material
    global Intersection* isect = isects + id;
    global BSDF* bsdf          = bsdfs + id;
    global TextureData* tex    = texData + id;

    if(bsdf->param.brdfType != EMITTER && bsdf->param.isTexture)
    {
          tex->hasBaseTex        = true;
          tex->materialID        = bsdf->materialID;            //to find image in CPU during texture lookup search in host code

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

__kernel void updateToTextureColorRT(
    global BSDF*         bsdfs,
    global TextureData*  texData
)
{
    //get thread id
    int id = get_global_id( 0 );

    //texture for pixel
    global TextureData* tex = texData + id;
    float4 texColor = getFloat4ARGB(tex->baseTexture.z);

    //get bsdf and parameter
    global BSDF* bsdf          = bsdfs + id;
    
    if(tex->hasBaseTex)
         bsdf->param.base_color     = texColor;
    

}

__kernel void fastShade(
    global Intersection* isects,
    global BSDF*         bsdfs,
    global int*          imageBuffer
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //default color
    float4 color = (float4)(0, 0, 0, 1);
    float4 color1 = (float4)(1, 1, 1, 1);
    
    //get intersect
    global Intersection* isect = isects + id;
    
    //get bsdf and parameter
    global BSDF* bsdf          = bsdfs + id;
    SurfaceParameter param     = bsdf->param;

    if(isect->hit)
    {
        float coeff = bsdf->localDirFix.z;
        color.xyz   = getQuickSurfaceColor(param, coeff).xyz;
        imageBuffer[id] = getIntARGB(color);
    }
}

__kernel void backgroundShade(
    global Intersection*      isects,
    global CameraStruct*      camera,
    global int*               imageBuffer,
    global Ray*               rays,

    global float4*            envmap,
    global EnvironmentGrid*   envgrid)
{
    //get thread id
    int id = get_global_id( 0 );

    //updated the intersected areas color
    global Intersection* isect = isects + id;
    global Ray* ray            = rays + id;

    if(!isect->hit)
    {
       //update
       if(envgrid->isPresent)
       {
            int envIndex = getSphericalGridIndex(envgrid->width, envgrid->height, ray->d);
            float4 envColor = envmap[envIndex];
            gammaFloat4(&envColor, 2.2f);
            imageBuffer[id] = getIntARGB(envColor);
       }
       else
            imageBuffer[id] = getIntARGB((float4)(0, 0, 0, 1));
    }
}

__kernel void updateGroupbufferShadeImage(
    global Intersection* isects,
    global CameraStruct* camera,
    global int* groupBuffer
)
{
    int id= get_global_id( 0 );

    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        groupBuffer[id] = getMaterial(isect->mat);
    }  
    else
        groupBuffer[id] = -1;
}