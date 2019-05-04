
//xor32 from Lecture 12 -  GPU Ray Tracing (2) by Jacco Bikker
float random_float(uint seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed * 2.3283064365387e-10f;
}

float2 random_float2(uint seed)
{
    float2 r;
    r.x = random_float(seed);
    r.y = random_float(seed);
    return r;
}
