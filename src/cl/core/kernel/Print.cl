#pragma OPENCL EXTENSION cl_amd_printf :enable

// print float value
void printFloat(float v)
{
    printf("%4.12f\n", v);
}

void printFloat2(float2 v)
{
    printf("%4.8v2f\n", v);
}

void printFloat3(float3 v)
{
    printf("%4.8v3f\n", v);
}

// print float4
void printFloat4(float4 v)
{
    printf("%4.8v4f\n", v);
}

// print int
void printInt(int i, bool newLine)
{
    if(newLine)
        printf("%2d\n", i);
    else
        printf("%2d  ", i);
}

void printlnInt(int i)
{
    printInt(i, true);
}

void printInt2(int2 v)
{
   printf("%d, %d\n", v.x, v.y);
}

void printInt4(int4 v)
{
   printf("%d, %d, %d, %d\n", v.x, v.y, v.z, v.w);
}

// print boolean
void printBoolean(bool value)
{        
   if(value)
      printf("true \n");
   else
      printf("false \n");
}
 
