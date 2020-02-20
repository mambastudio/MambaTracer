typedef struct
{
   float4 diffuse;              //diffuse    - r, g, b, w (pad)
   float diffuseWeight;         //diffuse    - diffuse weight

   float4 reflection;           //reflection - r, g, b, w (pad)
   float eu, ev, ior;           //reflection - eu, ev, ior
   bool iorEnabled;             //reflection - transmission enabled

   float4 emitter;              //emission   - r, g, b, w (power)
   bool emitterEnabled;         //emission   - emission enabled
}Material;

typedef struct
{
   int materialID;              //material id
   Frame frame;                 //local frame of reference
   float4 localDirFix;          //incoming (fixed) incoming direction, in local
}BSDF;

BSDF setupBSDF(global Ray* ray, global Intersection* isect)
{
   BSDF bsdf;
   bsdf.frame = get_frame(isect->n);
   //set local dir fix for ray incoming
   bsdf.localDirFix = local_coordinate(bsdf.frame, -ray->d);
   bsdf.materialID  = isect->mat;
   return bsdf;
}

float4 getMaterialColor(Material mat, float coeff)
{
   if(mat.emitterEnabled) return mat.emitter;
   else                   return mat.diffuse * coeff;
}

float4 getEmitterColor(Material mat)
{
   return mat.emitter * 2.f;
}

float4 sampledMaterialColor(Material mat)
{
   if(mat.emitterEnabled) return mat.emitter;
   else                   return mat.diffuse;
}

int selectBRDF(Material mat)
{
   if(mat.emitterEnabled) return 2;
   else                   return 0;
}

bool isEmitter(Material mat)
{
   return mat.emitterEnabled;
}


float pdfDiffuse(float4 localDirGen)
{
   return (localDirGen.z *(float)(M_1_PI));
}

float4 evaluateDiffuse(Material material, float4 localDirGen, float* directPdfW)
{
   *directPdfW = localDirGen.z * (float)(M_1_PI);
   return (float4)(material.diffuse.xyz * (float)(M_1_PI), 1.f);
}

float4 sampleDiffuse(Material material, float2 sample, float4* localDirGen, float* cosThetaGen, float* pdfW)
{
   *localDirGen = sample_hemisphere(sample);
   *pdfW        = pdfDiffuse(*localDirGen);
   *cosThetaGen = fabs(localDirGen->z);
   //printFloat4(material.diffuse);
   return (float4)(material.diffuse.xyz * (float)(M_1_PI), 1.f);
}

float4 evaluateBrdf(Material material, BSDF bsdf, float4 oWorldDirGen, float* oCosThetaGen, float* directPdfW)
{  
  //if not initialized with value, it has an issue (is it always in opencl?)
   float4 result = (float4)(0, 0, 0, 0);

   float4 localDirGen = local_coordinate(bsdf.frame, oWorldDirGen);
   if(localDirGen.z * bsdf.localDirFix.z < 0)
       return result;

   if(localDirGen.z < EPS_COSINE || bsdf.localDirFix.z < EPS_COSINE)
       return result;

   *oCosThetaGen = fabs(localDirGen.z);
   result = evaluateDiffuse(material, localDirGen, directPdfW);
   return result;
}

float4 sampleBrdf(Material material, BSDF bsdf, float2 sample, float4* oWorldDirGen, float* oCosThetaGen, float* pdfW)
{
   float4 localDirGen;
   float4 result = sampleDiffuse(material, sample, &localDirGen, oCosThetaGen, pdfW);
   *oWorldDirGen = world_coordinate(bsdf.frame, localDirGen);
   return result;
}
