#define FLOATMAX  3.402823e+38
#define FLOATMIN -3.402823e+38
#define HIT_MARKER 1
#define MISS_MARKER 0

// print float value
void printFloat(float v)
{
    printf("%4.8f\n", v);
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

void printInt2(int2 v)
{
   printf("%d, %d\n", v.x, v.y);
}

// print boolean
void printBoolean(bool value)
{
    printf(value ? "true \n" : "false \n");
}

// ray struct
typedef struct
{
   float4 o;
   float4 d;

   float4 inv_d;
   float tMin;
   float tMax;
   
   int4 sign;
   int2 extra;
}Ray;

// intersection
typedef struct
{
   float4 p;
   float4 n;
   float4 d;
   float2 uv;
   int mat;
   int id;
   int hit;
   float2 pixel;
}Intersection;

__kernel void test(global Intersection* isects, global Ray* rays)
{
  
}