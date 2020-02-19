#define SWAP(a,b) {__local int *tmp=a;a=b;b=tmp;}
#define INPUT_ISECT(index) (getDataIsect(index, length, isects))
#define getInput(index) (getData(index, length, data))
#define setOutput(index, value) (setData(index, value, length, data))

/*
  https://github.com/LariscusObscurus/HPC_Exercise/blob/master/src/kernels/scan.cl
*/

bool isIsectOkay(global Intersection* isect)
{
    return isect->hit && isect->sampled_brdf;
}

int getData(int index, global int* length, global int* data)
{
    if(index < *length)
        return data[index];
    else
        return 0;
}

int getDataIsect(int index, global int* length, global Intersection* isects)
{
    if(index < *length)
        return isIsectOkay(isects + index);
    else 
        return 0;
}

void setData(int index, int value, global int* length, global int* data)
{
    if(index < *length)
        data[index] = value;
}

int getDataF(int index, global float* data)
{
        return data[index];
}

void setDataF(int index, float value, global float* data)
{
        data[index] = value;
}


__kernel void initIntArray(
    global int* array)
{
    uint global_id = get_global_id(0);
    array[global_id] = 0;
}

__kernel void processIsectData(
    global Intersection* isects,
    global int* data)
{
    uint global_id   = get_global_id ( 0 );
    global Intersection* isect = isects + global_id;
    data[global_id]  = isIsectOkay(isect);
  
}
__kernel void  blelloch_scan_g(global int* data,
                               global int* group_sum,
                               global int* length,
                               local  int* temp)
{
     uint global_id = get_global_id ( 0 );
     uint local_id = get_local_id ( 0 );

     uint group_id = get_group_id ( 0 );
     uint group_size = get_local_size ( 0 );

     uint depth = 1 ;

     temp [local_id] = getInput(global_id);

     // upsweep   
     for ( uint stride = group_size >> 1 ; stride> 0 ; stride >>= 1 )
     {
          barrier (CLK_LOCAL_MEM_FENCE);
  
          if (local_id <stride)
          {
              uint i = depth * ( 2 * local_id + 1 ) - 1 ;
              uint j = depth * ( 2 * local_id + 2 ) - 1 ;
              temp [j] += temp [i];
          }
  
          depth <<= 1 ;
     }

     // set identity before downsweep
     if (local_id == 0 )
     {
        group_sum[group_id] = temp[group_size - 1];
        temp [group_size - 1] = 0 ;
     }

     // downsweep
     for (uint stride = 1 ; stride <group_size; stride <<= 1 ) {

        depth >>= 1 ;
        barrier (CLK_LOCAL_MEM_FENCE);

        if (local_id <stride) {
            uint i = depth * (2 * local_id + 1 ) - 1 ;
            uint j = depth * (2 * local_id + 2 ) - 1 ;

            int t = temp [j];
            temp [j] += temp [i];
            temp [i] = t;
        }
    }

    barrier (CLK_LOCAL_MEM_FENCE);
    setOutput(global_id, temp[local_id]);
}

__kernel void  blelloch_scan(global int* data,
                             global int* length,
                             local  int* temp)
{
     uint global_id = get_global_id ( 0 );
     uint local_id = get_local_id ( 0 );

     uint group_id = get_group_id ( 0 );
     uint group_size = get_local_size ( 0 );

     uint depth = 1 ;

     temp [local_id] = getInput(global_id);
     
     // upsweep   
     for ( uint stride = group_size >> 1 ; stride> 0 ; stride >>= 1 )
     {
          barrier (CLK_LOCAL_MEM_FENCE);
  
          if (local_id <stride) 
          {
              uint i = depth * ( 2 * local_id + 1 ) - 1 ;
              uint j = depth * ( 2 * local_id + 2 ) - 1 ;
              temp [j] += temp [i];
          }
  
          depth <<= 1 ;
     }

     // set identity before downsweep
     if (local_id == 0 )
        temp [group_size - 1] = 0 ;

     // downsweep
     for (uint stride = 1 ; stride <group_size; stride <<= 1 ) {

        depth >>= 1 ;
        barrier (CLK_LOCAL_MEM_FENCE);

        if (local_id <stride) {
            uint i = depth * (2 * local_id + 1 ) - 1 ;
            uint j = depth * (2 * local_id + 2 ) - 1 ;

            int t = temp [j];
            temp [j] += temp [i];
            temp [i] = t;
        }
    }

    barrier (CLK_LOCAL_MEM_FENCE);

    setOutput(global_id, temp[local_id]);
}

__kernel void add_groups(
    __global int* data,
    __global int* sums)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    data[global_id] = data[global_id] + sums[group_id];
}

__kernel void add_groups_n(
    __global int* data,
    __global int* sums,
    __global int* length)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if(global_id < *length)
        data[global_id] = data[global_id] + sums[group_id];
}

__kernel void  blelloch_scan_g_f(global float* data,
                                 global float* group_sum,
                                 local  float* temp)
{
     uint global_id = get_global_id ( 0 );
     uint local_id = get_local_id ( 0 );

     uint group_id = get_group_id ( 0 );
     uint group_size = get_local_size ( 0 );

     uint depth = 1 ;

     temp [local_id] = data[global_id];
     
     // upsweep   
     for ( uint stride = group_size >> 1 ; stride> 0 ; stride >>= 1 )
     {
          barrier (CLK_LOCAL_MEM_FENCE);
  
          if (local_id <stride) 
          {
              uint i = depth * ( 2 * local_id + 1 ) - 1 ;
              uint j = depth * ( 2 * local_id + 2 ) - 1 ;
              temp [j] += temp [i];
          }
  
          depth <<= 1 ;
     }

     // set identity before downsweep
     if (local_id == 0 )
     {
        group_sum[group_id] = temp[group_size - 1];
        temp [group_size - 1] = 0 ;
     }

     // downsweep
     for (uint stride = 1 ; stride <group_size; stride <<= 1 ) {

        depth >>= 1 ;
        barrier (CLK_LOCAL_MEM_FENCE);

        if (local_id <stride) {
            uint i = depth * (2 * local_id + 1 ) - 1 ;
            uint j = depth * (2 * local_id + 2 ) - 1 ;

            int t = temp [j];
            temp [j] += temp [i];
            temp [i] = t;
        }
    }

    barrier (CLK_LOCAL_MEM_FENCE);
    data[global_id] = temp[local_id];
}

__kernel void  blelloch_scan_f(global float* data,
                               local  float* temp)
{
     uint global_id = get_global_id ( 0 );
     uint local_id = get_local_id ( 0 );

     uint group_id = get_group_id ( 0 );
     uint group_size = get_local_size ( 0 );

     uint depth = 1 ;

     temp [local_id] = data[global_id];
     
     // upsweep   
     for ( uint stride = group_size >> 1 ; stride> 0 ; stride >>= 1 )
     {
          barrier (CLK_LOCAL_MEM_FENCE);
  
          if (local_id <stride) 
          {
              uint i = depth * ( 2 * local_id + 1 ) - 1 ;
              uint j = depth * ( 2 * local_id + 2 ) - 1 ;
              temp [j] += temp [i];
          }
  
          depth <<= 1 ;
     }

     // set identity before downsweep
     if (local_id == 0 )
        temp [group_size - 1] = 0 ;

     // downsweep
     for (uint stride = 1 ; stride <group_size; stride <<= 1 ) {

        depth >>= 1 ;
        barrier (CLK_LOCAL_MEM_FENCE);

        if (local_id <stride) {
            uint i = depth * (2 * local_id + 1 ) - 1 ;
            uint j = depth * (2 * local_id + 2 ) - 1 ;

            int t = temp [j];
            temp [j] += temp [i];
            temp [i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    data[global_id] = temp[local_id];
}
__kernel void add_groups_f(
    __global float* data,
    __global float* sums)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    data[global_id] = data[global_id] + sums[group_id];
}

__kernel void add_groups_n_f(
    __global float* data,
    __global float* sums,
    __global int* length)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if(global_id < *length)
        data[global_id] = data[global_id] + sums[group_id];
}

__kernel void total_f(__global float* buffer,
                      __global float* prefixsum,
                      __global int* length,
                      __global float* count)
{
    float i    = buffer[*length - 1];
    float j    = prefixsum[*length - 1];
    count[0] = i+j;
}

__kernel void totalElements_f(__global float* buffer,
                              __global float* prefixsum,
                              __global int* length,
                              __global float* count)
{
    float i = 0;
    if(buffer[*length - 1] > 0)
        i = 1;
    float j  = prefixsum[*length - 1];
    count[0] = i+j;
}

__kernel void copyFloatBuffer(__global float* buffer1,
                              __global float* buffer2)
{
    int global_id   = get_global_id(0);
    buffer1[global_id] = buffer2[global_id];
}

__kernel void copyDigitFloatBuffer(__global float* buffer1,
                                   __global float* buffer2)
{
    int global_id   = get_global_id(0);
    if(buffer2[global_id] > 0)
        buffer1[global_id] = 1;
    else
        buffer1[global_id] = 0;
}

__kernel void resetIntersection(global Intersection* isects)
{
    int global_id = get_global_id(0);

    global Intersection* isect = isects + global_id;
    isect->p             = (float4)(0, 0, 0, 0);
    isect->n             = (float4)(0, 0, 0, 0);
    isect->d             = (float4)(0, 0, 0, 0);
    isect->uv            = (float2)(0, 0);
    isect->mat           = 0;
    isect->sampled_brdf  = 0;
    isect->id            = 0;
    isect->hit           = 0;
    isect->throughput    = (float4)(1, 1, 1, 1);
    isect->pixel         = (float2)(0, 0);
    isect->hit           = MISS_MARKER;
    isect->mat           = -1;
}

__kernel void compactIntersection(__global Intersection* isects,
                                  __global Intersection* temp_isects,
                                  __global int* prefixsum)
{
    int global_id   = get_global_id(0);
    global Intersection* isect = isects + global_id;
    
    if(isIsectOkay(isect))
        temp_isects[prefixsum[global_id]] = *isect;
}

__kernel void transferIntersection(__global Intersection* isects,
                                   __global Intersection* temp_isects)
{
    int global_id   = get_global_id(0);
    isects[global_id] = temp_isects[global_id];
}
__kernel void totalIntersection(__global Intersection* isects,
                                __global int* prefixsum,
                                __global int* length,
                                __global int* count)
{
    global Intersection* isect = isects + (*length - 1);
    int i    = isIsectOkay(isect);
    int j    = prefixsum[*length - 1];
    count[0] = i+j;
}
