#define SWAP(a,b) {__local int *tmp=a;a=b;b=tmp;}
#define getInput(index) (getData(index, length, data))
#define INPUT_ISECT(index) (getDataIsect(index, length, isects))
#define setOutput(index, value) (setData(index, value, length, data))

/*
  https://github.com/LariscusObscurus/HPC_Exercise/blob/master/src/kernels/scan.cl
*/

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
        return isects[index].hit;
    else 
        return 0;
}

void setData(int index, int value, global int* length, global int* data)
{
    if(index < *length)
        data[index] = value;
}

__kernel void  blelloch_scan_isect_g(global Intersection* isects,
                                     global int* data,
                                     global int* group_sum,
                                     global int* length,
                                     local  int* temp)
{
     uint global_id = get_global_id ( 0 );
     uint local_id = get_local_id ( 0 );

     uint group_id = get_group_id ( 0 );
     uint group_size = get_local_size ( 0 );

     uint depth = 1 ;

     temp [local_id] = INPUT_ISECT(global_id);
     
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

__kernel void resetIntersection(global Intersection* isects)
{
    int global_id = get_global_id(0);

    Intersection defaultIsect;
    defaultIsect.hit = MISS_MARKER;
    defaultIsect.mat = -1;

    isects[global_id] = defaultIsect;
}

__kernel void compactIntersection(__global Intersection* isects,
                                  __global Intersection* temp_isects,
                                  __global int* prefixsum)
{
    int global_id   = get_global_id(0);
    global Intersection* isect = isects + global_id;
    
    if(isect->hit)
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
     int i    = isects[*length - 1].hit;
     int j    = prefixsum[*length - 1];
     count[0] = i+j;

}