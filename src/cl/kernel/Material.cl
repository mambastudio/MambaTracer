enum bsdf_type {DIFFUSE, EMITTER};

typedef struct
{
   //this surface is done by texture
    bool            isTexture;
    int             numTexture;
    
    //what brdf
    int             brdfType;
            
    //brdf parameters
    float4          base_color;
    float           diffuse_roughness;
    float4          specular_color;
    float           specular_IOR;
    float           metalness;
    float           transmission;
    float4          transmission_color;
    float           emission;
    float4          emission_color;
    float           anisotropy_nx;
    float           anisotropy_ny;
  
}SurfaceParameter;

typedef struct
{
    SurfaceParameter param1;
    SurfaceParameter param2;
    
    bool             isPortal;

    int              opacityType;
    float            fresnel;
    float            opacity;
  
}Material;

typedef struct
{
   int materialID;              //material id

   bool isPortal;
   
   int paramLevel;
   SurfaceParameter  param;     //chosen surface

   Frame frame;                 //local frame of reference
   float4 localDirFix;          //incoming (fixed) incoming direction, in local
}BSDF;


typedef struct
{
   int4 baseTexture;                               //x, y-coord, argb  //actual texture color
   int4 opacityTexture;                            //x, y-coord, argb  //opacity texture
   int  hasOpacity, hasBaseTex, materialID, paramLevel;
}TextureData;

BSDF setupBSDF(global Ray* ray, global Intersection* isect, global Material* materials)//, TextureData texdata, State state
{
   BSDF bsdf;
   bsdf.frame = get_frame(isect->n);

   //set local dir fix for ray incoming
   bsdf.localDirFix = local_coordinate(bsdf.frame, -ray->d);
   
   //set material id
   bsdf.materialID  = isect->mat;
   
   //initialize other parameters
   bsdf.isPortal =  materials[bsdf.materialID].isPortal;
   
   //choose layer parameter
   bsdf.param = materials[bsdf.materialID].param1;     //future it will be decided stochastically
   bsdf.paramLevel = 0;
   
   //return
   return bsdf;
}

BSDF setupBSDFAreaLight(global Material* materials, TriangleMesh mesh, int triangleIndex)
{
   BSDF bsdf;
   global Material* material = materials + getMaterial(mesh.faces[triangleIndex].mat);
   bsdf.param = material->param1;     //first parameter is the only one to have emission.
   return bsdf;
}

float4 getQuickSurfaceColor(SurfaceParameter param, float coeff)
{
   if(param.brdfType == EMITTER)
   {
       return param.emission_color;
   }
   else if(param.brdfType == DIFFUSE)
   {
       return param.base_color * coeff;
   }
   else
       return (float4)(0, 0, 0, 1);
}

float4 getMaterialColor(Material mat, float coeff)
{
   SurfaceParameter param = mat.param1;
   if(param.brdfType == EMITTER)
   {
       return param.emission_color;
   }
   else if(param.brdfType == DIFFUSE)
   {
       return param.base_color * coeff;
   }
   else
       return (float4)(0, 0, 0, 1);
}

float pdfDiffuse(float4 localDirGen)
{
   return (localDirGen.z *(float)(M_1_PI));
}

float4 evaluateDiffuse(BSDF bsdf, float4 localDirGen, float* directPdfW)
{
   *directPdfW = localDirGen.z * (float)(M_1_PI);
   return (float4)(bsdf.param.base_color.xyz * (float)(M_1_PI), 1.f);
}

float4 sampleDiffuse(BSDF bsdf, float2 sample, float4* localDirGen, float* cosThetaGen, float* pdfW)
{
   *localDirGen = sample_hemisphere(sample);
   *pdfW        = pdfDiffuse(*localDirGen);
   *cosThetaGen = fabs(localDirGen->z);
   //printFloat4(material.diffuse);
   return (float4)(bsdf.param.base_color.xyz * (float)(M_1_PI), 1.f);
}

float4 evaluateBrdf(BSDF bsdf, float4 oWorldDirGen, float* oCosThetaGen, float* directPdfW)
{  
  //if not initialized with value, it has an issue (is it always in opencl?)
   float4 result = (float4)(0, 0, 0, 0);

   float4 localDirGen = local_coordinate(bsdf.frame, oWorldDirGen);
   if(localDirGen.z * bsdf.localDirFix.z < 0)
       return result;

   if(localDirGen.z < EPS_COSINE || bsdf.localDirFix.z < EPS_COSINE)
       return result;

   *oCosThetaGen = fabs(localDirGen.z);
   result = evaluateDiffuse(bsdf, localDirGen, directPdfW);
   return result;
}

float4 sampleBrdf(BSDF bsdf, float2 sample, float4* oWorldDirGen, float* oCosThetaGen, float* pdfW)
{
   float4 localDirGen, result;
   result = sampleDiffuse(bsdf, sample, &localDirGen, oCosThetaGen, pdfW);
   *oWorldDirGen = world_coordinate(bsdf.frame, localDirGen);
   return result;
}
