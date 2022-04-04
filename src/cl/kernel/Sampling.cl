#define PI 3.14159265358979323846f
typedef struct
{
   int2 seed;
   float  frameCount;
}State;

typedef struct
{
   int seed;
   float frameCount;
}State2;

/// Hash function
unsigned int WangHash(int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

int initRNG(global State2* state)
{
  
}  

//unique seed for each thread 
//https://github.com/jbikker/lighthouse2/blob/master/lib/RenderCore_Optix7Filter/optix/.optix.cu
int2 generate_seed(global State* state)
{
    int global_id = get_global_id(0);

    int2 seed;
    seed.x = ( global_id  + (int)(state->frameCount)  * (state->seed.x));
    seed.y = ( global_id  + (int)(state->frameCount)  * (state->seed.y));
    return seed;
}

//https://github.com/straaljager/OpenCL-path-tracing-tutorial-3-Part-2/blob/master/opencl_kernel.cl
float get_random(int2* state)
{
    /* hash the seeds */
    state->x = WangHash(state->x);
    state->y = WangHash(state->y);

    unsigned int ires = ((state->x) << 16) + (state->y);

    /* use union struct to convert int to float */
    union {
	float f;
	unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
    return (res.f - 2.0f) / 2.0f;
}

float2 random_float2(int2* state)
{
    float2 r;
    r.x = get_random(state);
    r.y = get_random(state);  
    return r;
}

float3 random_float3(int2* state)
{
    float3 r;
    r.x = get_random(state);
    r.y = get_random(state);  
    r.z = get_random(state);
    return r;
}

float4 random_float4(int2* state)
{
    float4 r;
    r.x = get_random(state);
    r.y = get_random(state);  
    r.z = get_random(state);
    r.w = get_random(state);
    return r;
}

//Refer to https://stackoverflow.com/questions/363681/how-do-i-generate-random-integers-within-a-specific-range-in-java
//between 0 (inclusive) to range (exclusive)
int2 random_int2_range(int2* state, int rangeX, int rangeY)
{
    int2 ri;
    float2 rf  = random_float2(state);
    ri.x       = (int)(rf.x * rangeX);
    ri.y       = (int)(rf.y * rangeY);
    return ri;   
}

//between 0 (inclusive) to range (exclusive)
int random_int_range(int2* state, int range)
{
    float rf  = get_random(state);
    return (int)(rf * range);
}

int select_int_range(float rf, int range)
{
    return (int)(rf * range);
}

float4 sample_hemisphere(
    float2 sample
)
{
    float r1 = sample.x;
    float r2 = sample.y;

    
    float term1 = 2.f * M_PI * r1;
    float term2 = sqrt(1.f - r2);
    
    float x = cos(term1) * term2;
    float y = sin(term1) * term2;
    float z = sqrt(r2);
    
    return (float4)(x, y, z, 0);
}

float4 SampleCosHemisphereW(
    float2                  sample,
    float                   *pdfW)
{
    float term1 = 2.f * M_PI * sample.x;
    float term2 = sqrt(1.f - sample.y);

    float4 ret  = (float4)(
                           cos(term1) * term2,
                           sin(term1) * term2,
                           sqrt(sample.y), 
                           0);

    if(pdfW)
    {
        *pdfW = ret.z * M_1_PI;
    }

    return ret;
}

float4 SamplePowerCosHemisphereW(
    float2  sample,
    float   power,
    float   *pdfW)
{
    float term1 = 2.f * M_PI * sample.x;
    float term2 = pow(sample.y, 1.f / (power + 1.f));
    float term3 = sqrt(1.f - term2 * term2);

    if(pdfW)
    {
        *pdfW = (power + 1.f) * pow(term2, power) * (0.5f * M_1_PI);
    }

    return (float4)(
        cos(term1) * term3,
        sin(term1) * term3,
        term2, 0);
}

float PowerCosHemispherePdfW(
    float4  n,
    float4  v,
    float   power)
{
    float cosTheta = fmax(0.f, dot(n, v));

    return (power + 1.f) * pow(cosTheta, power) * (M_1_PI * 0.5f);
}


// returns barycentric coordinates
float2 sample_barycentric(float2 samples)
{
    float term = (float) sqrt(samples.x);
    return (float2)(1.f - term, samples.y * term);
}

//Sample Triangle
float4 sample_triangle(float2 samples, float4 p1, float4 p2, float4 p3)
{
    //float4 e1 = p2 - p1;
    //float4 e2 = p3 - p1;
    
    //uv
    //float2 uv = sample_barycentric(samples);
    
    //return point
    //return p1 + e1 * uv.x + e2 * uv.y;
    float alpha = 1 - sqrt(samples.x);
    float beta  = (1 - samples.y)*sqrt(samples.x);
    float gamma = samples.y * sqrt(samples.x);
    return p1*alpha + p2*beta + p3*gamma;
}

// Mis power (1 for balance heuristic)
float mis(float aPdf)
{
    return aPdf;
}

// Mis weight for 2 pdfs
float mis2(float aSamplePdf, float aOtherPdf)
{
    return mis(aSamplePdf) / (mis(aSamplePdf) + mis(aOtherPdf));
}

// Utilities for converting PDF between Area (A) and Solid angle (W)
// WtoA = PdfW * cosine / distance_squared
// AtoW = PdfA * distance_squared / cosine

float pdfWtoA(
    float aPdfW,
    float aDist,
    float aCosThere)
{
    return aPdfW * fabs(aCosThere) / (aDist*aDist);
}

float pdfAtoW(
    float aPdfA,
    float aDist,
    float aCosThere)
{
    return aPdfA * (aDist*aDist) / fabs(aCosThere);
}
