

__kernel void InitIntData(global int* intData)
{
    int id= get_global_id( 0 );   
    intData[id] = 0; 
}

__kernel void InitIntData_1(global int* intData)
{
    int id= get_global_id( 0 );
    intData[id] = -1;
}

__kernel void initIntDataRGB(global int* intData)
{

    int id= get_global_id( 0 );
    float4 f = (float4)(0, 0, 0, 1);
    intData[id] = getIntARGB(f);
}

__kernel void InitFloatData(global float* floatData)
{
    int id= get_global_id( 0 );   
    floatData[id] = 0;
}

__kernel void initFloat4DataXYZW(global float4* float4Data)
{
    int id = get_global_id(0);
    float4Data[id] = (float4)(0, 0, 0, 0);
}

__kernel void initFloat4DataXYZ(global float4* float4Data)
{
    int id = get_global_id(0);
    float4Data[id] = (float4)(0, 0, 0, 1);
}

__kernel void initPixelIndices(global int* pixel_indices)
{
    int id = get_global_id(0);
    pixel_indices[id] = id;  
}

void InitIsect(global Intersection* isect)
{
    isect->p             = (float4)(0, 0, 0, 0);
    isect->n             = (float4)(0, 0, 0, 0);
    isect->d             = (float4)(0, 0, 0, 0);
    isect->uv            = (float2)(0, 0);
    isect->sampled_brdf  = 0;
    isect->id            = 0;
    isect->throughput    = (float4)(1, 1, 1, 1);
    isect->pixel         = (float2)(0, 0);
    isect->hit           = MISS_MARKER;
    isect->mat           = -1;
}

__kernel void initIntersection(global Intersection* isects)
{
    int global_id = get_global_id(0);
    global Intersection* isect = isects + global_id;
    
    InitIsect(isect);
}

/*
    - Essential Mathematics for Games and Interactive Applications: 2nd edition, pg 203
    - Soon to expand for thin lens, which is trivial

    - Notice we do many operations on global memory level to avoid private memory cache strain
      or it will render glitches as it came to my understanding
*/

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

    //set pixel & active
    r->pixel = pixel;
    r->extra.x = true;
}

