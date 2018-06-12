#define DELTA(i,j) delta(sortedMortonCodes,num_prims,i,j)

int delta(global int* sortedMortonCodes, int num_prims, int i1, int i2)
{
    // Select left end
    int left = min(i1, i2);
    // Select right end
    int right = max(i1, i2);
    // This is to ensure the node breaks if the index is out of bounds
    if (left < 0 || right >= num_prims)
    {            
        return -1;
    }
    // Fetch Morton codes for both ends
    int left_code = sortedMortonCodes[left];
    int right_code = sortedMortonCodes[right];

    // Special handling of duplicated codes: use their indices as a fall
    return left_code != right_code ? clz(left_code ^ right_code) : (32 + clz(left ^ right));
}

int findSplit(global int* sortedMortonCodes, int num_prims, int first, int end)
{
    // Fetch codes for both ends
    int left = first;
    int right = end;

    // Calculate the number of identical bits from higher end
    int num_identical = DELTA(left, right);
    
    do
    {
        // Proposed split
        int new_split = (right + left) / 2;

        // If it has more equal leading bits than left and right accept it
        if (DELTA(left, new_split) > num_identical)
            left = new_split;            
        else            
            right = new_split;            
    }
    while (right > left + 1);
    
    return left;
}

int2 findSpan(global int* sortedMortonCodes, int num_prims, int idx)
{
    // Find the direction of the range
    int d = sign((float)(DELTA(idx, idx+1) - DELTA(idx, idx-1)));

    // Find minimum number of bits for the break on the other side
    int delta_min = DELTA(idx, idx-d);
    
    // Search conservative far end
    int lmax = 2;
    while (DELTA(idx,idx + lmax * d) > delta_min)
        lmax *= 2;
    
    // Search back to find exact bound
    // with binary search
    int l = 0;
    int t = lmax;
    do
    {
        t /= 2;
        if(DELTA(idx, idx + (l + t)*d) > delta_min)
        {
            l = l + t;
        }
    }
    while (t > 1);
            
    // Pack span 
    int2 span;
    span.x = min(idx, idx + l*d);
    span.y = max(idx, idx + l*d);

    return span;
}

__kernel void testMortons(global int* mortonPrimitive, global int* numOfPrims)
{
     int id= get_global_id( 0 );
     int num_prims = numOfPrims[0];
     
     int2 range =  findSpan(mortonPrimitive, num_prims, id);
     int split = findSplit(mortonPrimitive, num_prims, range.x, range.y);
     
     for(int i = 0; i<num_prims; i++)
        printInt(mortonPrimitive[i], true);
     //printInt2(range);
     //printInt(split, true);
}
