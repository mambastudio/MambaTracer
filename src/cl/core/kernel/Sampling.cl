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

//https://github.com/straaljager/OpenCL-path-tracing-tutorial-3-Part-2/blob/master/opencl_kernel.cl
float get_random(unsigned int *seed0, unsigned int *seed1) 
{
    /* hash the seeds */
    *seed0 = WangHash(*seed0);
    *seed1 = WangHash(*seed1);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    /* use union struct to convert int to float */
    union {
	float f;
	unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
    return (res.f - 2.0f) / 2.0f;
}

float2 random_float2(unsigned int *randSeed0, unsigned int *randSeed1)
{
    float2 r;
    r.x = get_random(randSeed0, randSeed1);
    r.y = get_random(randSeed1, randSeed0);
    return r;
}

//Refer to https://stackoverflow.com/questions/363681/how-do-i-generate-random-integers-within-a-specific-range-in-java
int2 random_int2_range(unsigned int* seed0, unsigned int* seed1, int rangeX, int rangeY)
{
    int2 ri;
    float2 rf  = random_float2(seed0, seed1);
    ri.x       = (int)(rf.x * rangeX);
    ri.y       = (int)(rf.y * rangeY);
    return ri;   
}

float4 sample_hemisphere(
    float2 sample
)
{
    float r1 = sample.x;
    float r2 = sample.y;
    
    float term1 = 2.f * PI * r1;
    float term2 = sqrt(1.f - r2);
    
    float x = cos(term1) * term2;
    float y = sin(term1) * term2;
    float z = sqrt(r2);
    
    return (float4)(x, y, z, 0);
}
