//intersection predicate
bool isIsectOkay(global Intersection* isect)
{
    return isect->hit && isect->sampled_brdf;
}

//init predicate and prefixsum
__kernel void initIsectPredicateAndPrefixSum(
    global Intersection* isects,
    global int* predicate,
    global int* prefixsum)
{
    uint global_id   = get_global_id ( 0 );
    global Intersection* isect = isects + global_id;

    predicate[global_id]  = isIsectOkay(isect);
    prefixsum[global_id]  = predicate[global_id];
}

//we are actually scattering here intersection
__kernel void compactIntersection(__global Intersection* isects,
                                  __global Intersection* temp_isects,
                                  __global int* predicate,
                                  __global int* prefixsum)
{
    int global_id   = get_global_id(0);
    global Intersection* isect = isects + global_id;

    if(predicate[global_id])
        temp_isects[prefixsum[global_id]] = *isect;
}

__kernel void transferIntersection(__global Intersection* isects,
                                   __global Intersection* temp_isects)
{
    int global_id   = get_global_id(0);
    isects[global_id] = temp_isects[global_id];
}

//we are actually scattering here rays
__kernel void compactRays(__global Ray* rays,
                          __global Ray* temp_rays,
                          __global int* predicate,
                          __global int* prefixsum)
{
    int global_id   = get_global_id(0);
    global Ray* ray = rays + global_id;

    if(predicate[global_id])
        temp_rays[prefixsum[global_id]] = *ray;
}

__kernel void transferRays(__global Intersection* rays,
                           __global Intersection* temp_rays)
{
    int global_id   = get_global_id(0);
    rays[global_id] = temp_rays[global_id];
}

//we are actually scattering pixels here
__kernel void compactPixels(__global int* pixels,
                          __global int* temp_pixels,
                          __global int* predicate,
                          __global int* prefixsum)
{
    int global_id   = get_global_id(0);
    global int* pixel = pixels + global_id;

    if(predicate[global_id])
        temp_pixels[prefixsum[global_id]] = *pixel;
}

__kernel void transferPixels(__global int* pixels,
                           __global int* temp_pixels)
{
    int global_id   = get_global_id(0);
    pixels[global_id] = temp_pixels[global_id];
}



