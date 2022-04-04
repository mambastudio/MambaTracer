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
   //data for whole environment map
   global float4* envmap;
   global float* lum;
   global float* sat;

   //environment map size (whole environment map)
   SATRegion region;
   
   //directional sampling to find tile to sample light path
   global LightGrid* lightGrid;
   
   //from sampling and temporary kept here to insert in the light grid
   float luminance;
   int lightGridIndex;

}EnvironmentLight;

void accumLightGrid(EnvironmentLight aLight, float luminance, int lightGridIndex)
{
   atomicAdd(&aLight.lightGrid->accum[lightGridIndex], luminance);
}

/**
* LIGHT FUNCTIONS
*/
AreaLight getAreaLight(Bsdf bsdf, TriangleMesh mesh, int triangleIndex)
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
   light.intensity    = bsdf.param.emission_color * bsdf.param.emission_param.y;

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

EnvironmentLight getEnvironmentLight(global LightGrid* lightGrid, global float4* envmap, global float* lum, global float* sat)
{
     SATRegion region;
     setRegion(&region, 0, 0, lightGrid->width, lightGrid->height);
     region.nu = lightGrid->width;
     region.nv = lightGrid->height;
     EnvironmentLight envlight = {envmap, lum, sat, region, lightGrid};
     return envlight;
}


float4 illuminateEnvironmentLight(
       EnvironmentLight aLight,
       float4           aReceivingPosition,
       float4           aSample,
       float4           *oDirectionToLight,
       float            *oDistance,
       float            *oDirectPdfW,
       float            *luminance,
       int              *lightGridIndex
)
{
  int offsetLL[2];
  float pdfLL[1];
  float uvLL[2];
  
  int offsetSS[2];
  float pdfSS[1];
  float uvSS[2];
  


  //sample tile
  int subgridIndex = subgridIndexFromCamera(aLight.lightGrid, aReceivingPosition);
  sampleSubgridContinuous(aLight.lightGrid, subgridIndex, aSample.x, aSample.y, uvLL, offsetLL, pdfLL);

  //get unit bound of tile withing cell
  float4 unitBound    = getSubgridUnitBound(aLight.lightGrid, offsetLL);

  //sat region based on tile bound
  SATRegion tileSAT   = getSubRegionFromUnitBound(aLight.region, unitBound);

  //sample within sat region   
  sampleContinuous(tileSAT, aSample.z, aSample.w, uvSS, offsetSS, pdfSS, aLight.sat, aLight.lum);

  *oDirectionToLight  = getSphericalDirection(uvSS[0], uvSS[1]);

  int index           = getSphericalGridIndex(aLight.lightGrid->width, aLight.lightGrid->height, *oDirectionToLight);
  float4 contrib      = aLight.envmap[index];
  contrib.xyz         = contrib.xyz;
  
  float sinTheta = sin(uvSS[1] * M_PI);

  *oDirectPdfW = pdfSS[0] /(2 * M_PI * M_PI * sin(uvSS[1] * M_PI));
  *oDistance = FLOATMAX;
 
  if(sinTheta == 0)
     *oDirectPdfW = 0;

  //calculate light grid accumulations
  *luminance       = Luminance(contrib.xyz);
  *lightGridIndex  = globalIndexInSubgrid(aLight.lightGrid, subgridIndex, offsetLL[0], offsetLL[1]);

  return contrib;
}

float4 illuminateEnvironmentLight1(
       EnvironmentLight aLight,
       float4           aReceivingPosition,
       float4           aSample,
       float4           *oDirectionToLight,
       float            *oDistance,
       float            *oDirectPdfW,
       float            *luminance,
       int              *lightGridIndex
)
{
    int offset[2];
    float pdf[1];
    float uv[2];
    
    sampleContinuous(aLight.region, aSample.x, aSample.y, uv, offset, pdf, aLight.sat, aLight.lum);
    
    *oDirectionToLight  = getSphericalDirection(uv[0], uv[1]);
  
    int index           = getSphericalGridIndex(aLight.lightGrid->width, aLight.lightGrid->height, *oDirectionToLight);
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
    //light grid index based on hit point if any
    int subgridIndex    = subgridIndexFromCamera(aLight.lightGrid, aHitPoint);
    
    //index in tiles of subgrid
    int2 tileXY         = tileIndexXYFromCamera(aLight.lightGrid, aRayDirection);
    int offsetLL[2]     = {tileXY.x, tileXY.y};

    //get unit bound of tile withing cell
    float4 unitBound    = getSubgridUnitBound(aLight.lightGrid, offsetLL);

    //sat region based on tile bound
    SATRegion tileSAT   = getSubRegionFromUnitBound(aLight.region, unitBound);
    
    //calculate pdf in sat region
    int2 satXY          = getSphericalGridXY(aLight.lightGrid->width, aLight.lightGrid->height, aRayDirection);
    float pdfSS         = getPdfSAT(tileSAT, satXY.x, satXY.y, aLight.sat, aLight.lum);
    int offsetSS[2]     = {satXY.x, satXY.y};
    
    //overall pdf
    float s             = offsetSS[1]/(float)(lengthY(aLight.region));
    *oDirectPdfA        = pdfSS /(2 * M_PI * M_PI * sin(s * M_PI));

    //get contribution and return
    int index           = getSphericalGridIndex(aLight.lightGrid->width, aLight.lightGrid->height, aRayDirection);
    float4 contrib      = aLight.envmap[index];
    
    //calculate light grid accumulations
    float luminance       = Luminance(contrib.xyz);
    int   lightGridIndex  = globalIndexInSubgrid(aLight.lightGrid, subgridIndex, offsetLL[0], offsetLL[1]);
    accumLightGrid(aLight, luminance, lightGridIndex);

    return contrib;
}

float4 getRadianceEnvironmentLight1(
       EnvironmentLight aLight,
       float4           aRayDirection,
       float4           aHitPoint,
       float            *oDirectPdfA
)
{
    int2 xy             = getSphericalGridXY(aLight.lightGrid->width, aLight.lightGrid->height, aRayDirection);
    int index           = getSphericalGridIndex(aLight.lightGrid->width, aLight.lightGrid->height, aRayDirection);
    float4 contrib      = aLight.envmap[index];
    float pdfXY         = getPdfSAT(aLight.region, xy.x, xy.y, aLight.sat, aLight.lum);
    float v             = xy.y/(float)(lengthY(aLight.region));
    *oDirectPdfA        = pdfXY /(2 * M_PI * M_PI * sin(v * M_PI));

    return contrib;
  
}