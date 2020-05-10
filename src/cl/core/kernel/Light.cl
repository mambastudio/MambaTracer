// triangle face indices
typedef struct
{
   float4     p1;
   float4     e1;
   float4     e2;
   Frame      frame;
   float4     intensity;
   float      invArea;
}AreaLight;

AreaLight getAreaLight(global Material* materials, TriangleMesh mesh, int triangleIndex)
{
   AreaLight light;
   
   global Material* material = materials + getMaterial(mesh.faces[triangleIndex].mat);

   float4 p1 = getP1(mesh, triangleIndex);
   float4 p2 = getP2(mesh, triangleIndex);
   float4 p3 = getP3(mesh, triangleIndex);
   
   float4 n;
   if(hasNormals(mesh, triangleIndex))
   {
      float4 n1 = getN1(mesh, triangleIndex);
      float4 n2 = getN2(mesh, triangleIndex);
      float4 n3 = getN3(mesh, triangleIndex);

      n = normalize((n1 + n2 + n3)/3.f);
   }
   else
      n  = getNormal(p1, p2, p3);
      
   float3 e1 = p2.xyz - p1.xyz;
   float3 e2 = p3.xyz - p1.xyz;
    
   float3 normal      = cross(e1, e2);
   float len          = length(normal);
   
   light.invArea      = 2.f/len;
   light.frame        = get_frame(n);
   light.p1           = p1;
   light.e1           = (float4)(e1, 0);
   light.e2           = (float4)(e2, 0);
   light.intensity    = getEmitterColor(*material);
      
   return light;
}

float4 sampleAreaLight(float2 samples, AreaLight aLight)
{
    float2 uv = sample_barycentric(samples);
    return aLight.p1 + aLight.e1 * uv.x + aLight.e2 * uv.y;
}

float4 illuminateAreaLight(
       AreaLight aLight,
       float4    aReceivingPosition,
       float2    aSample,
       float4    *oDirectionToLight,
       float     *oDistance,
       float     *oDirectPdfW)
{
    float4 lightpoint       = sampleAreaLight(aSample, aLight);
    *oDirectionToLight      = lightpoint - aReceivingPosition;
    float distSqr           = dot(*oDirectionToLight, *oDirectionToLight);
    *oDistance              = sqrt(distSqr);
    *oDirectionToLight      = (*oDirectionToLight) / (*oDistance);

    float cosNormalDir      = dot(aLight.frame.mZ, -(*oDirectionToLight));

    // too close to, or under, tangent
    if(cosNormalDir < EPS_COSINE)
    {
         return (float4)(0.f);
    }

    *oDirectPdfW = aLight.invArea * distSqr / cosNormalDir;
    return aLight.intensity;
}

float4 getRadianceAreaLight(
       AreaLight aLight,
       float4    aRayDirection,
       float4    aHitPoint,
       float     *oDirectPdfA
)
{
       float cosOutL         = max(0.f, dot(aLight.frame.mZ, -aRayDirection));
       
       if(cosOutL == 0)
            return (float4)(0);
            
       if(oDirectPdfA)
            *oDirectPdfA = aLight.invArea;
            
       return aLight.intensity;
}

