
//moller ray-triangle intersection
float fastTriangleIntersection(Ray r, float4 p1, float4 p2, float4 p3)
{
     float4 e1, e2, h, s, q;
     float a, f, b1, b2;

     e1 = p2 - p1;
     e2 = p3 - p1;
     h  = cross(r.d, e2);
     a  = dot(e1, h);

     if (a > -0.0000001 && a < 0.0000001)
        return r.tMax;

     f  = 1/a;

     s  = r.o - p1;
     b1 = f * dot(s, h);
     q  = cross(s, e1);
     b2 = f * dot(r.d, q);
     
     float t  = f * dot(e2, q);
     
     if (b1 < 0.0 || b1 > 1.0 || b2 < 0.0 || b1 + b2 > 1.0 ||  t < 0.f || t > r.tMax)
        return r.tMax;
     else
        return f * dot(e2, q);
}

float2 triangleBarycentrics(float4 p, float4 p1, float4 p2, float4 p3)
{ 
    //since w is not initialized (nan) hence why we use xyz (some drivers - intel - handle uninitialized variables as nan).
    float3 e1 = p2.xyz - p1.xyz;
    float3 e2 = p3.xyz - p1.xyz;
    float3  e = p.xyz  - p1.xyz;
    float d00 = dot(e1, e1);
    float d01 = dot(e1, e2);
    float d11 = dot(e2, e2);
    float d20 = dot(e, e1);
    float d21 = dot(e, e2);

    float denom = (d00 * d11 - d01 * d01);

    if (denom == 0.f)
        return (float2)(0.f, 0.f);

    float const invdenom = 1.f / denom;
    float const b1 = (d11 * d20 - d01 * d21) * invdenom;
    float const b2 = (d00 * d21 - d01 * d20) * invdenom;

    return (float2)(b1, b2);
}

//https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
float2 triangleBarycentricsFromUVMesh(float4 p, float4 p1, float4 p2, float4 p3, float2 uv1, float2 uv2, float2 uv3)
{
    // calculate vectors from point p to vertices p1, p2 and p3:
    float3 f1 = p1.xyz - p.xyz;
    float3 f2 = p2.xyz - p.xyz;
    float3 f3 = p3.xyz - p.xyz;
    
    // calculate the areas and factors (order of parameters doesn't matter):
    float a      = length(cross(p1.xyz - p2.xyz, p1.xyz - p3.xyz)); // main triangle area a
    float a1     = length(cross(f2, f3)) / a; // p1's triangle area / a
    float a2     = length(cross(f3, f1)) / a; // p2's triangle area / a
    float a3     = length(cross(f1, f2)) / a; // p3's triangle area / a
    
    float2 uv    = uv1 * a1 + uv2 * a2 + uv3 * a3;
    return uv;
}

float triangleInverseArea(float4 p1, float4 p2, float4 p3)
{
    float3 e1 = p2.xyz - p1.xyz;
    float3 e2 = p3.xyz - p1.xyz;
    
    float3 normal = cross(e1, e2);
    float len     = length(normal);
    return 2.f/len;  
}
