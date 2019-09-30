#define PI 3.14159265358979323846f

/// Hash function
uint WangHash(int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

//xor32 from Lecture 12 -  GPU Ray Tracing (2) by Jacco Bikker
float random_float(int* seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed * 2.3283064365387e-10f;
}

float2 random_float2(int* seed)
{
    float2 r;
    r.x = random_float(seed);
    r.y = random_float(seed);
    return r;
}

//Refer to https://stackoverflow.com/questions/363681/how-do-i-generate-random-integers-within-a-specific-range-in-java
int2 random_int2_range(int* seed, int rangeX, int rangeY)
{
    int2 ri;
    float2 rf  = random_float2(seed);
    ri.x       = (int)(rf.x * rangeX);
    ri.y       = (int)(rf.y * rangeY);
    return ri;   
}

float4 sample_hemisphere(
    float2 sample
)
{
    float r1 = sample.y;
    float r2 = 1 - r1; //sample.y;
    
    float term1 = 2.f * PI * r1;
    float term2 = sqrt(1.f - r2);
    
    float x = cos(term1) * term2;
    float y = sin(term1) * term2;
    float z = sqrt(r2);
    
    return (float4)(x, y, z, 0);
}
