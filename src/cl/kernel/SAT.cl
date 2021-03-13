typedef struct
{
    //full region
    int nu;
    int nv;
    
    //sub region
    int minX, minY;
    int strideX, strideY; //exclusive since they represent length
  
}SATRegion;

void setRegion(SATRegion* region, int minX, int minY, int strideX, int strideY)
{
    if(minX < 0 || minY < 0)
        return;
    if(minX + strideX > region->nu || minY + strideY > region->nv)
        return;
    if(strideX <= 0 || strideY <= 0)
        return;
    region->minX = minX;
    region->minY = minY;
    region->strideX = strideX;
    region->strideY = strideY;
}


int lengthX(SATRegion region)
{
    return region.strideX;
}

int lengthY(SATRegion region)
{
    return region.strideY;
}

int getLastIndexX(SATRegion region)
{
    return region.minX + lengthX(region) - 1;
}

int getLastIndexY(SATRegion region)
{
    return region.minY + lengthY(region) - 1;
}

int getRegionArea(SATRegion region)
{
    return lengthX(region)*lengthY(region);
}

void setMin(SATRegion* region, int minX, int minY)
{
    setRegion(region, minX, minY, region->strideX, region->strideY);
}

void setStride(SATRegion* region, int strideX, int strideY)
{
    setRegion(region, region->minX, region->minY, strideX, strideY);
}

int getSATIndex(SATRegion region, int x, int y)
{
    if(x < 0 || y < 0)
        return -1;
    if(x > lengthX(region)-1 || y > lengthY(region) - 1)
        return -1;
    return (y + region.minY) * region.nu + (x + region.minX) ; //y*width + x
}

void initRegion(SATRegion* region, int nu, int nv)
{
    setRegion(region, 0, 0, nu, nv);
}

float getFunc(SATRegion region, int x, int y, global float* func)
{
    int index = getSATIndex(region, x, y);
    return func[index];
}

float getCummulative(SATRegion region, int x, int y, global float* sat)
{
    int index = getSATIndex(region, x, y);
    return sat[index];
}

void setCummulative(SATRegion region, int x, int y, float value, global float* sat)
{
    int index = getSATIndex(region, x, y);
    sat[index] = value;
}

//inclusive scan
void prefixRowSAT(SATRegion region, int y, global float* sat)
{       
    float val = 0;
    for(int x = 0; x<lengthX(region); x++)
    {
        val = val + getCummulative(region, x, y, sat);
        setCummulative(region, x, y, val, sat);
    }
}

//inclusive scan
void prefixColSAT(SATRegion region, int x, global float* sat)
{
    float val = 0;
    for(int y = 0; y<lengthY(region); y++)
    {
        val = val + getCummulative(region, x, y, sat);
        setCummulative(region, x, y, val, sat);   
    }
}

float getValueSAT(SATRegion region, int x, int y, global float* sat)
{
    if(x < 0)
        return 0;
    if(y < 0)
        return 0;
    if(x > lengthX(region) - 1)
        return 0;
    if(y > lengthY(region) - 1)
        return 0;        
    else
        return getCummulative(region, x, y, sat);
}

float getSATRange(SATRegion region, int minX, int minY, int maxX, int maxY, global float* sat)
{
    float A1 = getValueSAT(region, maxX, maxY, sat);
    float B1 = getValueSAT(region, minX-1, maxY, sat);
    float C1 = getValueSAT(region, maxX, minY-1, sat);
    float D1 = getValueSAT(region, minX-1, minY-1, sat);
    return A1+D1-B1-C1;
}

float getFuncIntConditional(SATRegion region, int y, global float* sat)
{
    /**
     * getSATRange(0, y, region.getLastIndexX(), y)
     * - this function calculates cdf in one line row along rowY
     * - functional integral conditional, you divide by the length of line
     */
    return getSATRange(region, 0, y, getLastIndexX(region), y, sat)/lengthX(region);
}
    
float getConditional(SATRegion region, int x, int y, global float* sat) //columnX, rowY
{
    float funcInt = getFuncIntConditional(region, y, sat);
    return getSATRange(region, 0, y, x - 1, y, sat)/lengthX(region)/funcInt;
}

float getMarginal(SATRegion region, int y, global float* sat) //rowY
{
    float marginalLast = getValueSAT(region, getLastIndexX(region), getLastIndexY(region), sat);
    return getValueSAT(region, getLastIndexX(region), y - 1, sat)/marginalLast;
}

float getPdfContinuousConditional(SATRegion region, int x, int y, global float* sat, global float* func) //offset along col, which row y
{
    float funcValue = getFunc(region, x, y, func);
    float funcInt = getFuncIntConditional(region, y, sat);
    
    return funcValue/funcInt;
}

float getPdfContinuousMarginal(SATRegion region, int y, global float* sat)
{
    float funcInt = getFuncIntConditional(region, y, sat);
    float lastSAT = getValueSAT(region, getLastIndexX(region), getLastIndexY(region), sat);
    
    return (funcInt * getRegionArea(region)) / lastSAT;
}

/*
Branching in opencl is quite slow, hence one has to avoid it if possible
To avoid it in binary search, use the select method, an inbuilt function of opencl
*/

//https://stackoverflow.com/questions/24989455/is-a-binary-search-a-good-fit-for-opencl
//branchless
int upperBoundConditional(SATRegion region, int y, int first, int last, float value, global float* sat)
{

    int begin = first;
    int end = last;

    while(begin != end) {
        int mid = begin + (end - begin) / 2;
        float midValue =  getConditional(region, mid, y, sat);

        bool b_right = !(value < midValue);
        begin = select(begin, (mid + 1), b_right);
        end = select(mid, end, b_right); // c ? b : a
    }

    return begin;

}

//https://stackoverflow.com/questions/24989455/is-a-binary-search-a-good-fit-for-opencl
//branchless
int upperBoundMarginal(SATRegion region, int first, int last, float value, global float* sat)
{
    int begin = first;
    int end = last;

    while(begin != end) {
        int mid = begin + (end - begin) / 2;
        float midValue =  getMarginal(region, mid, sat);

        bool b_right = !(value < midValue);
        begin = select(begin, (mid + 1), b_right);
        end = select(mid, end, b_right); // c ? b : a
    }

    return begin;
}

float sampleContinuousConditional(SATRegion region, float u, int y, int* off, float* pdf, global float* sat, global float* func)
{
    int ptr = upperBoundConditional(region, y, 0, lengthX(region), u, sat); //linear search
    int offset = max(0, ptr - 1);
    
    //set offset
    off[0] = offset;
    
    // Compute offset along CDF segment
    float du = (u - getConditional(region, offset, y, sat)) / (getConditional(region, offset + 1, y, sat) - getConditional(region, offset, y, sat));
   
    // Compute PDF for sampled offset
    pdf[0] = getPdfContinuousConditional(region, offset, y, sat, func);

    return (offset + du) /lengthX(region);
}

float sampleContinuousMarginal(SATRegion region, float u, int* off, float* pdf, global float* sat)
{
    int ptr = upperBoundMarginal(region, 0, lengthY(region), u, sat); //linear search
    int offset = max(0, ptr - 1);

    //set offset
    off[0] = offset;
    
    // Compute offset along CDF segment
    float du = (u - getMarginal(region, offset, sat)) / (getMarginal(region, offset + 1, sat) - getMarginal(region, offset, sat));
    
    // Compute PDF for sampled offset
    pdf[0] = getPdfContinuousMarginal(region, offset, sat);
           
    return (offset + du) / lengthY(region);
}


//executed by kernal(1, 1) meaning it's linear execution without parallel invocation and it's called once
void calculateSAT(SATRegion region, global float* sat)
{
    for(int i = 0; i< lengthY(region); i++)
        prefixRowSAT(region, i, sat);
    for(int i = 0; i< lengthX(region); i++)
        prefixColSAT(region, i, sat);
}

//2D Dimension Sampling
void sampleContinuous(SATRegion region, float u0, float u1, float* uv, float* pdf, global float* sat, global float* func)
{
    float pdfs[2];
    float pdfTemp[1];
    int v[1];
    int vtemp[1];
    
    //start with marginal and then conditional
    uv[1] = sampleContinuousMarginal(region, u1, v, pdfTemp, sat);
    pdfs[1] = pdfTemp[0];
    uv[0] = sampleContinuousConditional(region, u0, v[0], vtemp, pdfTemp, sat, func);
    pdfs[0] = pdfTemp[0];
    

      
    //overall pdf   
    pdf[0] = pdfs[0] * pdfs[1];
}

float getPdfSAT(SATRegion region, int x, int y, global float* sat, global float* func)
{
    float pdfV = getPdfContinuousMarginal(region, y, sat);
    float pdfU = getPdfContinuousConditional(region, x, y, sat, func); //offset along col, which row y
    
    return pdfU * pdfV;
}