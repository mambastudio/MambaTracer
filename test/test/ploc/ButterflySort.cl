BVHNode getNode(int index, int length, global BVHNode* nodes)
{
    BVHNode node;
    node.mortonCode = INT_MAX;
    if(index >= length)
        return node;
    else 
        return nodes[index];
}

bool isGreaterThan(int posStart, int posEnd, int length, global BVHNode* nodes)
{
    int value1 = getNode(posStart, length, nodes).mortonCode;
    int value2 = getNode(posEnd, length, nodes).mortonCode;

    return value1 > value2;
}

void swapNodes(int PosSIndex, int PosEIndex, global BVHNode* nodes)
{       
    BVHNode tmp          = nodes[PosSIndex];
    nodes[PosSIndex]     = nodes[PosEIndex];
    nodes[PosEIndex]     = tmp;
}

__kernel void butterfly1(global BVHNode* nodes, global int* lengthSize, global float* powerX)
{
     int gid = get_global_id(0);

     int t = gid;
     int radix = 2;
     int length = lengthSize[0];
     int PowerX = powerX[0];

     int yIndex      = (int) (t/(PowerX/radix));
     int kIndex      = (int) (t%(PowerX/radix));
     int PosStart    = (int) (kIndex + yIndex * PowerX);
     int PosEnd      = (int) (PowerX - kIndex - 1 + yIndex * PowerX);

     if(isGreaterThan(PosStart, PosEnd, length, nodes))
         swapNodes(PosStart, PosEnd, nodes);
}

__kernel void butterfly2(global BVHNode* nodes, global int* lengthSize, global float* powerX)
{
    int gid = get_global_id(0);

    int t = gid;
    int radix = 2;
    int length = lengthSize[0];
    int PowerX = powerX[0];

    int yIndex      = (int) (t/(PowerX/radix));
    int kIndex      = (int) (t%(PowerX/radix));
    int PosStart    = (int) (kIndex + yIndex * PowerX);
    int PosEnd      = (int) (kIndex + yIndex * PowerX + PowerX/radix);
                            
    if(isGreaterThan(PosStart, PosEnd, length, nodes))
        swapNodes(PosStart, PosEnd, nodes);
}