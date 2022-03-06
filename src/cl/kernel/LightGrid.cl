/*
Based on "Adaptive Environment Sampling on CPU and GPU"

The SAT here doesn't adopt the subregion SAT as mentioned in the paper,
and this is due to optimisation purpose in calculating SAT on the fly.

*/

typedef struct
{
    //is present
    bool isPresent;
    
    //sample tile
    int sampleTile;
    
    //to remove later
    int       width;
    int       height;

    //full region
    int nu, nv;
    int subgridRangeX, subgridRangeY; //same as tile -> local size   (16, 32)

    //camera position is important to know which light grid we are suppose to sample
    float4 cameraPosition;

    //16 * 32 * lightGrid(100 * 50)
    float accum[2560000];
    
    //16 * 32 * lightGrid(100 * 50)
    float func[2560000];

    //16 * 32 * lightGrid(100 * 50)
    float sat[2560000];
    
}LightGrid;

bool isSubgridIndexInRangeXY(global LightGrid* lightGrid, int sx, int sy)
{
    bool val1 = select(true, false, sx >= lightGrid->subgridRangeX || sx < 0);
    bool val2 = select(true, false, sy >= lightGrid->subgridRangeY || sy < 0);
    return val1&&val2;
}

//number of subgrid in x direction
int subgridCountX(global LightGrid* lightGrid)
{
    return lightGrid->nu/lightGrid->subgridRangeX;
}

//number of subgrid in y direction
int subgridCountY(global LightGrid* lightGrid)
{
    return lightGrid->nv/lightGrid->subgridRangeY;
}

int getSubgridLastIndexX(global LightGrid* lightGrid)
{
    return lightGrid->subgridRangeX-1;
}

int getSubgridLastIndexY(global LightGrid* lightGrid)
{
    return lightGrid->subgridRangeY-1;
}

int subgridIndexFromCamera(global LightGrid* lightGrid, float4 hitPoint)
{
    int cellWidth                   = lightGrid->nu/lightGrid->subgridRangeX;
    int cellHeight                  = lightGrid->nv/lightGrid->subgridRangeY;
    
    float4 cameraDir                = normalize(hitPoint - lightGrid->cameraPosition); //camera direction

    int subgridIndex                = getSphericalGridIndex(cellWidth, cellHeight, cameraDir);   //subgrid or cell  (direction from camera)
    
    return subgridIndex;
}

int2 tileIndexXYFromCamera(global LightGrid* lightGrid, float4 direction)
{
    int tileIndex                  = getSphericalGridIndex(lightGrid->subgridRangeX, lightGrid->subgridRangeY, direction);
    
    int2 tileXY;

    tileXY.x = tileIndex % lightGrid->subgridRangeX;
    tileXY.y = tileIndex / lightGrid->subgridRangeX;
    
    return tileXY;
}

//total number of subgrid
int subgridCount(global LightGrid* lightGrid)
{
    return subgridCountX(lightGrid)*subgridCountY(lightGrid);
}

//area of one subgrid
int subgridArea(global LightGrid* lightGrid)
{
    return lightGrid->subgridRangeX*lightGrid->subgridRangeY;
}

int subgridGlobalIndex2(global LightGrid* lightGrid, int subgridIndexX, int subgridIndexY)
{
    return subgridIndexX * lightGrid->subgridRangeX + subgridIndexY * subgridArea(lightGrid) * subgridCountX(lightGrid);
}

int subgridGlobalIndex1(global LightGrid* lightGrid, int subgridIndex)
{
    int subgridIndexX = subgridIndex%subgridCountX(lightGrid);
    int subgridIndexY = subgridIndex/subgridCountX(lightGrid);
    
    return subgridGlobalIndex2(lightGrid, subgridIndexX, subgridIndexY);
}

int2 subgridGlobalIndexXY(global LightGrid* lightGrid, int subgridIndex)
{
    int globalIndex = subgridGlobalIndex1(lightGrid, subgridIndex);

    int globalIndexX  = globalIndex%lightGrid->nu;
    int globalIndexY  = globalIndex/lightGrid->nv;
    
    return (int2)(globalIndexX, globalIndexY);
}

int2 globalIndexXYInSubgrid(global LightGrid* lightGrid, int subgridIndex, int localX, int localY)
{
    int2 globalXY = subgridGlobalIndexXY(lightGrid, subgridIndex);
    globalXY.x += localX;
    globalXY.y += localY;
    return globalXY;
}

int globalIndexInSubgrid(global LightGrid* lightGrid, int subgridIndex, int localX, int localY)
{
    int2 globalXY = globalIndexXYInSubgrid(lightGrid, subgridIndex, localX, localY);
    return globalXY.x + globalXY.y*lightGrid->nu;
}

//get sat value within specific subgrid
float getSubgridValueSAT(global LightGrid* lightGrid, int subgridIndex, int sx, int sy)
{
    bool isSubgridInRange = isSubgridIndexInRangeXY(lightGrid, sx, sy);
    float value           = select(0.f, lightGrid->sat[globalIndexInSubgrid(lightGrid, subgridIndex, sx, sy)], isSubgridInRange);

    return value;
}

float getSubgridSATRange(global LightGrid* lightGrid, int subgridIndex, int minX, int minY, int maxX, int maxY)
{
    float A1 = getSubgridValueSAT(lightGrid, subgridIndex, maxX, maxY);
    float B1 = getSubgridValueSAT(lightGrid, subgridIndex, minX-1, maxY);
    float C1 = getSubgridValueSAT(lightGrid, subgridIndex, maxX, minY-1);
    float D1 = getSubgridValueSAT(lightGrid, subgridIndex, minX-1, minY-1);
    return A1+D1-B1-C1;
}

//(xmin, ymin, xmax, ymax) of range (0,1)
float4 getSubgridUnitBound(global LightGrid* lightGrid, int* offset)    //subIndex range([0-1], [0-1])
{
    int2 tileXY      = (int2)(offset[0], offset[1]);     //make sure the tile index is known
    float2 unitMin   = (float2)(tileXY.x/(float)lightGrid->subgridRangeX, tileXY.y/(float)lightGrid->subgridRangeY);     //calculate the min xy based on tile index
    float2 unitMax   = (float2)((tileXY.x+1)/(float)lightGrid->subgridRangeX, (tileXY.y+1)/(float)lightGrid->subgridRangeY);
    float4 unitBound = (float4)(unitMin.x, unitMin.y,
                                unitMax.x, unitMax.y);
    return unitBound;
}

//PROBABILITIES

float getSubgridFuncIntConditional(global LightGrid* lightGrid, int subgridIndex, int y)
{        
    return getSubgridSATRange(lightGrid, subgridIndex, 0, y, getSubgridLastIndexX(lightGrid), y);
}

float getSubgridFunc(global LightGrid* lightGrid, int subgridIndex, int sx, int sy)
{
    int index = globalIndexInSubgrid(lightGrid, subgridIndex, sx, sy);
    return lightGrid->func[index];
}

float getSubgridConditional(global LightGrid* lightGrid, int subgridIndex, int x, int y) //columnX, rowY
{
    float funcInt = getSubgridFuncIntConditional(lightGrid, subgridIndex, y);
    return getSubgridSATRange(lightGrid, subgridIndex, 0, y, x - 1, y)/funcInt;
}

float getSubgridMarginal(global LightGrid* lightGrid, int subgridIndex, int y) //rowY
{
    float marginalLast = getSubgridValueSAT(lightGrid, subgridIndex, getSubgridLastIndexX(lightGrid), getSubgridLastIndexY(lightGrid));
    return getSubgridValueSAT(lightGrid, subgridIndex, getSubgridLastIndexX(lightGrid), y - 1)/marginalLast;
}

float getSubgridPdfContinuousConditional(global LightGrid* lightGrid, int subgridIndex, int x, int y) //offset along col, which row y
{
    float funcValue = getSubgridFunc(lightGrid, subgridIndex, x, y);
    float funcInt = getSubgridFuncIntConditional(lightGrid, subgridIndex, y);
    
    return funcValue/funcInt;
}
float getSubgridPdfContinuousMarginal(global LightGrid* lightGrid, int subgridIndex, int y)
{
    float funcInt = getSubgridFuncIntConditional(lightGrid, subgridIndex, y);
    float lastSAT = getSubgridValueSAT(lightGrid, subgridIndex, getSubgridLastIndexX(lightGrid), getSubgridLastIndexY(lightGrid));

    return (funcInt * subgridArea(lightGrid)) / lastSAT;
}

//https://stackoverflow.com/questions/24989455/is-a-binary-search-a-good-fit-for-opencl
//branchless
int upperSubgridBoundConditional(global LightGrid* lightGrid, int subgridIndex, int y, int first, int last, float value)
{

    int begin = first;
    int end = last;

    while(begin != end) {
        int mid = begin + (end - begin) / 2;
        float midValue =  getSubgridConditional(lightGrid, subgridIndex, mid, y);

        bool b_right = !(value < midValue);
        begin = select(begin, (mid + 1), b_right);
        end = select(mid, end, b_right); // c : b ? a
    }

    return begin;

}

//https://stackoverflow.com/questions/24989455/is-a-binary-search-a-good-fit-for-opencl
//branchless
int upperSubgridBoundMarginal(global LightGrid* lightGrid, int subgridIndex, int first, int last, float value)
{
    int begin = first;
    int end = last;

    while(begin != end) {
        int mid = begin + (end - begin) / 2;
        float midValue =  getSubgridMarginal(lightGrid, subgridIndex, mid);    

        bool b_right = !(value < midValue);
        begin = select(begin, (mid + 1), b_right);
        end = select(mid, end, b_right); // c ? b : a
    }
    return begin;
}

float sampleSubgridContinuousConditional(global LightGrid* lightGrid, int subgridIndex, float u, int y, int* off, float* pdf)
{
    int ptr = upperSubgridBoundConditional(lightGrid, subgridIndex, y, 0, lightGrid->subgridRangeX, u); //linear search
    int offset = max(0, ptr - 1);
    
    //set offset
    off[0] = offset;
    
    // Compute offset along CDF segment
    float du = (u - getSubgridConditional(lightGrid, subgridIndex, offset, y)) /
                   (getSubgridConditional(lightGrid, subgridIndex, offset + 1, y) -
                    getSubgridConditional(lightGrid, subgridIndex, offset, y));

    // Compute PDF for sampled offset
    pdf[0] = getSubgridPdfContinuousConditional(lightGrid, subgridIndex, offset, y);

    return (offset + du) /lightGrid->subgridRangeX;
}

int sampleSubgridDiscreteConditional(global LightGrid* lightGrid, int subgridIndex, float u, int y, float* pdf)
{
    int ptr = upperSubgridBoundConditional(lightGrid, subgridIndex, y, 0, lightGrid->subgridRangeX, u); //linear search
    int offset = max(0, ptr - 1);
  
    // Compute PDF for sampled offset
    pdf[0] = getSubgridPdfContinuousConditional(lightGrid, subgridIndex, offset, y);
    
    return offset;
}

float sampleSubgridContinuousMarginal(global LightGrid* lightGrid, int subgridIndex, float u, int* off, float* pdf)
{
    int ptr = upperSubgridBoundMarginal(lightGrid, subgridIndex, 0, lightGrid->subgridRangeY, u); //linear search
    int offset = max(0, ptr - 1);

    //set offset
    off[0] = offset;

    // Compute offset along CDF segment
    float du = (u - getSubgridMarginal(lightGrid, subgridIndex, offset)) /
                   (getSubgridMarginal(lightGrid, subgridIndex, offset + 1) -
                    getSubgridMarginal(lightGrid, subgridIndex, offset));
    
    // Compute PDF for sampled offset
    pdf[0] = getSubgridPdfContinuousMarginal(lightGrid, subgridIndex, offset);
           
    return (offset + du) / lightGrid->subgridRangeY;
}

int sampleSubgridDiscreteMarginal(global LightGrid* lightGrid, int subgridIndex, float u, float* pdf)
{
    int ptr = upperSubgridBoundMarginal(lightGrid, subgridIndex, 0, lightGrid->subgridRangeY, u); //linear search
    int offset = max(0, ptr - 1);
    
    // Compute PDF for sampled offset
    pdf[0] = getSubgridPdfContinuousMarginal(lightGrid, subgridIndex, offset);
    
    return offset;
}

//2D Dimension Sampling
void sampleSubgridContinuous(__global LightGrid* lightGrid, int subgridIndex, float u0, float u1, float* uv, int* offset, float* pdf)
{
    float pdfs[2];

    //start with marginal and then conditional
    uv[1] = sampleSubgridContinuousMarginal(lightGrid, subgridIndex, u1, offset + 1, pdfs + 1);
    uv[0] = sampleSubgridContinuousConditional(lightGrid, subgridIndex, u0, offset[1], offset, pdfs);

    //overall pdf
    *pdf = pdfs[0] * pdfs[1];
}

//2D Dimension Sampling
void sampleSubgridDiscrete(__global LightGrid* lightGrid, int subgridIndex, float2 u, int* offset, float* pdf)
{
    float pdfs[2];
    float pdfTemp[1];
    
    //start with marginal and then conditional
    offset[1]   = sampleSubgridDiscreteMarginal(lightGrid, subgridIndex, u.y, pdfTemp);
    pdfs[1]     = pdfTemp[0];
    offset[0]   = sampleSubgridDiscreteConditional(lightGrid, subgridIndex, u.x, offset[1], pdfTemp);
    pdfs[0]     = pdfTemp[0];
                    
    pdf[0] = pdfs[0] * pdfs[1]/subgridArea(lightGrid);
}

float getSubgridPdfContinuous(__global LightGrid* lightGrid, int subgridIndex, int x, int y)
{
    float pdfV = getSubgridPdfContinuousMarginal(lightGrid, subgridIndex, y);
    float pdfU = getSubgridPdfContinuousConditional(lightGrid, subgridIndex, x, y); //offset along col, which row y
    
    return pdfU * pdfV;
}

//aux = rangeX = local size
//Kogge-Stone prefix sum
__kernel void prefixSumRowSubgrid(__global LightGrid* lightGrid, __local float* aux)
{
    int idl  = get_local_id(0); // index in workgroup
    int idg  = get_global_id(0);
    int idgr = get_group_id(0);
    int lSize = get_local_size(0);

    aux[idl] = lightGrid->func[idg];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);  //ensure read to local first

    for(int offset = 1; offset < lSize; offset *= 2)
    {
         private float temp; //take note on the float
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
__kernel void prefixSumColSubgrid(__global LightGrid* lightGrid, __local float* aux)
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
         private float temp;    //take note on the float
         if(idl >= offset) temp = aux[idl - offset];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
         if(idl >= offset) aux[idl] = temp + aux[idl];
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    //casting to int reduces sampling errors as highlighted in the paper
    lightGrid->sat[index] = aux[idl]; //read back to the SAT grid
}

//global is 2560000, local is 32 (or 64, 128, 256)
//to make it faster, one can init local temp array and read back to global
__kernel void initLightGrid(__global LightGrid* lightGrid)
{
    int idg  = get_global_id(0);
    lightGrid->accum[idg] = 0;
    lightGrid->func[idg]  = 0;
    lightGrid->sat[idg]  = 0;
}

__kernel void initFuncAndSat(__global  LightGrid* lightGrid)
{
    int idg  = get_global_id(0);
    lightGrid->func[idg]  = 0;
    lightGrid->sat[idg]  = 0;
}

__kernel void calculateFunc(__global LightGrid* lightGrid)
{
    int idg  = get_global_id(0);
    lightGrid->func[idg] = lightGrid->accum[idg];
}

/*
This kernels downwards are for testing purpose on proof of concept/debugging.
They are not used anywhere in the main path tracer.
*/

//globalsize 1, localsize 1
__kernel void  sampleSubgridTest(__global LightGrid* lightGrid, __global int* subgridIndex, __global float2* u, __global float2* uv, __global float* pdf)
{
    float2  uvTp;
    float   pdfTp;

    //sampleSubgridContinuous(lightGrid, *subgridIndex, *u, &uvTp, NULL, &pdfTp);

    *uv = uvTp;
    *pdf = pdfTp;

}