
__kernel void TransferIntersection(__global Intersection* isects,
                                   __global Intersection* temp_isects)
{
    int global_id   = get_global_id(0);
    isects[global_id] = temp_isects[global_id];
}

__kernel void TransferPixels(__global int* pixels,
                           __global int* temp_pixels)
{
    int global_id   = get_global_id(0);
    pixels[global_id] = temp_pixels[global_id];
}


__kernel void CompactAtomic(global Intersection* isects,
                            global Intersection* tempIsects,
                            global int* pixels,
                            global int* tempPixels,
                            global int* count)
{
     int global_id             = get_global_id (0);
     
     global Intersection* isect = isects + global_id;
     if(isect->hit)
     {
          int index = atomic_inc(count);
          tempIsects[index] = isects[global_id];
          tempPixels[index] = pixels[global_id];
     }
}

