__kernel void InitIntDataRGB(global int* intData)
{

    int id= get_global_id( 0 );
    float4 f = (float4)(0, 0, 0, 1);
    intData[id] = getIntARGB(f);
}   

void InitIsect(global Intersection* isect)
{
    isect->p             = (float4)(0, 0, 0, 0);
    isect->n             = (float4)(0, 0, 0, 0);
    isect->d             = (float4)(0, 0, 0, 0);
    isect->uv            = (float2)(0, 0);
    isect->id            = -1;
    isect->hit           = MISS_MARKER;
    isect->mat           = -1;
}

__kernel void InitIntersection(global Intersection* isects)
{
    int global_id = get_global_id(0);
    global Intersection* isect = isects + global_id;
    
    InitIsect(isect);
}

__kernel void InitIntDataToIndex(global int* intData)
{
    int id= get_global_id( 0 );   
    intData[id] = id; 
}

__kernel void InitIntData(
    global int* array)
{
    uint global_id = get_global_id(0);
    array[global_id] = 0;
}


__kernel void InitFloat4DataXYZ(global float4* float4Data)
{
    int id = get_global_id(0);
    float4Data[id] = (float4)(0, 0, 0, 1);
}

__kernel void InitEnvironmentLum(global float4* envmap, global float* lum, global float* lumsat, global EnvironmentGrid* envgrid)
{
    //global id and pixel making
    int id= get_global_id( 0 );
    int size = envgrid->width * envgrid->height;

    if(id < size)
    {
        lum[id] = Luminance(envmap[id].xyz);
        lumsat[id] = lum[id];
    }
}

//globalsize = 1, localsize = 1
__kernel void InitEnvironmentSAT(global EnvironmentGrid* envgrid, global float* sat)
{
    SATRegion region;
    setRegion(&region, 0, 0, envgrid->width, envgrid->height);    
    region.nu = envgrid->width;
    region.nv = envgrid->height;
    calculateSAT(region, sat);
}

__kernel void InitCameraRayDataJitter(
    global CameraStruct* camera,
    global Ray* rays,
    global State* state)
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
    
    //seeds for each thread
    int2 seed = generate_seed(state);
    
    //generate random number (0 to 1)
    float2 sample                = random_float2(&seed);
    float jitter1                = 1.f/camera->dimension.x * (2 * sample.x - 1.f);
    float jitter2                = 1.f/camera->dimension.y * (2 * sample.y - 1.f);

    //direction (px, py, pz, 0) and origin (0, 0, 0, 0)
    r->d = normalize((float4)(a * (2.f * pixel.x/camera->dimension.x - 1 + jitter1), -2.f * pixel.y/camera->dimension.y + 1 + jitter2, -d, 0));
    r->o = 0;  //will be important for global illumination, when we reuse the rays

    //transform to world space
    r->o = transform_point4(camera_matrix.mInv, r->o);
    r->d = transform_vector4(camera_matrix.mInv, r->d);

    //init ray
    initGlobalRay(r, r->o, r->d);
}
