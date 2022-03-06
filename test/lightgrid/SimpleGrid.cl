typedef struct
{
    int nu, nv;

   //16 * 16
    float func[256];

    //16 * 16
    float sat[256];
  
}SimpleGrid;


//aux = rangeX = local size
//Kogge-Stone prefix sum
__kernel void prefixSumRowSubgrid(__global SimpleGrid* lightGrid, __local float* aux)
{
    int idl  = get_local_id(0); // index in workgroup
    int idg  = get_global_id(0);
    int idgr = get_group_id(0);
    int lSize = get_local_size(0);

    aux[idl] = lightGrid->func[idg];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);  //ensure read to local first

    for(int offset = 1; offset < lSize; offset *= 2)
    {
         private float temp;  //make sure if it's int, your dealing with ints and vice versa
         if(idl >= offset) temp = aux[idl - offset];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
         if(idl >= offset) aux[idl] = temp + aux[idl];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    lightGrid->sat[idg] = aux[idl];     //read back to the SAT grid
}

//aux = rangeY = local size
//Kogge-Stone prefix sum
__kernel void prefixSumColSubgrid(__global SimpleGrid* lightGrid, __local float* aux)
{
    int idl  = get_local_id(0); // index in workgroup
    int idg  = get_global_id(0);
    int idgr = get_group_id(0);
    int lSize = get_local_size(0);
    
    int i = lSize * idgr;
    int xi = i/lightGrid->nv; //get col index
    int yi = i%lightGrid->nv + idl;   //get row index
    
    int index = xi + yi*lightGrid->nu; //global index of array

    aux[idl] = lightGrid->sat[index];    //read now from SAT (prefix row has been done already from previous kernel)
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);  //read to local first
    
    for(int offset = 1; offset < lSize; offset *= 2)
    {
         private float temp; //make sure if it's int, your dealing with ints and vice versa
         if(idl >= offset) temp = aux[idl - offset];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
         if(idl >= offset) aux[idl] = temp + aux[idl];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    //casting to int reduces sampling errors as highlighted in the paper
    lightGrid->sat[index] = aux[idl]; //read back to the SAT grid
}