/**
* LIGHT TYPES
*/
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

// environment map
typedef struct
{
   global float4* envmap;
   global float* lum;
   global float* sat;
   //is it the whole env map or section
   SATRegion region;
   //env map details (in future it will be the sampling distribution too)
   global EnvironmentGrid* envgrid;
   
   //for sampling
   float pdf[1];
   float uv[2];
   float indexU;
   float indexV;
}EnvironmentLight;

typedef struct
{
   int type;

   AreaLight areaLight;
   EnvironmentLight envLight;
  
}Light;

/**
* LIGHT FUNCTIONS
*/
AreaLight getAreaLight(BSDF bsdf, TriangleMesh mesh, int triangleIndex)
{
   AreaLight light;

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
   light.intensity    = bsdf.param.emission_color * bsdf.param.emission;
      
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

EnvironmentLight getEnvironmentLight(global EnvironmentGrid* envgrid, global float4* envmap, global float* lum, global float* sat)
{
     SATRegion region;
     setRegion(&region, 0, 0, envgrid->width, envgrid->height);
     region.nu = envgrid->width;
     region.nv = envgrid->height;
     EnvironmentLight envlight = {envmap, lum, sat, region, envgrid};
     return envlight;
}

int sampleEnvironmentLight(float2 samples, EnvironmentLight* aLight)
{
    sampleContinuous(aLight->region, samples.x, samples.y, &aLight->uv, &aLight->pdf, aLight->sat, aLight->envmap);
    int uIndex     = (int)(aLight->uv[0] * aLight->envgrid->width);
    int vIndex     = (int)(aLight->uv[1] * aLight->envgrid->height);
    aLight->indexU = uIndex;
    aLight->indexV = vIndex;
    int index      = (int)(uIndex + vIndex * aLight->envgrid->width);
    //printFloat(index);
    return index;
}

float4 illuminateEnvironmentLight(
       EnvironmentLight aLight,
       float4           aReceivingPosition,
       float2           aSample,
       float4           *oDirectionToLight,
       float            *oDistance,
       float            *oDirectPdfW
)
{
  float uv[2];
  float pdf[1];
  sampleContinuous(aLight.region, aSample.x, aSample.y, &uv, &pdf, aLight.sat, aLight.lum);

  *oDirectionToLight  = getSphericalDirection(uv[0], uv[1]);
  int index           = getSphericalGridIndex(aLight.envgrid->width, aLight.envgrid->height, *oDirectionToLight);
  float4 contrib      = aLight.envmap[index];
  contrib.xyz         = contrib.xyz;

  float sinTheta = sin(uv[1] * M_PI);

  *oDirectPdfW = pdf[0] /(2 * M_PI * M_PI * sin(uv[1] * M_PI));
  *oDistance = FLOATMAX;
  
  if(sinTheta == 0)
     *oDirectPdfW = 0;

  return contrib;
}

float4 getRadianceEnvironmentLight(
       EnvironmentLight aLight,
       float4           aRayDirection,
       float4           aHitPoint,
       float            *oDirectPdfA
)
{
    int2 xy             = getSphericalGridXY(aLight.envgrid->width, aLight.envgrid->height, aRayDirection);
    int index           = getSphericalGridIndex(aLight.envgrid->width, aLight.envgrid->height, aRayDirection);
    float4 contrib      = aLight.envmap[index];
    float pdfXY         = getPdfSAT(aLight.region, xy.x, xy.y, aLight.sat, aLight.lum);
    float v             = xy.y/(float)(lengthY(aLight.region));
    *oDirectPdfA        = pdfXY /(2 * M_PI * M_PI * sin(v * M_PI));

    return contrib;
}