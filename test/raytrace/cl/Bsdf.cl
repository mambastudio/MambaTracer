#define EPS_PHONG 1e-3f

typedef struct
{
    float diffProb;
    float glossyProb;
    float reflProb;
    float refrProb;
}ComponentProbabilities;

enum Events
{
    kNONE        = 0,
    kDiffuse     = 1,
    kGlossy      = 2,
    kReflect     = 4,
    kRefract     = 8,
    kEmit        = 12,
    kSpecular    = (kReflect  | kRefract),
    kNonSpecular = (kDiffuse  | kGlossy),
    kAll         = (kSpecular | kNonSpecular)
};

typedef struct
{
   //this surface is done by texture
   //be careful with boolean, it's not mapped properly by custom java struct from CPU depending on hardware
    int            isDiffuseTexture;
    int            isGlossyTexture;
    int            isRoughnessTexture;
    int            isMirrorTexture;

    //brdf parameters
    float4          diffuse_color;
    float4          diffuse_param;  //x = scale
    float4          glossy_color;
    float4          glossy_param;   //x = scale, y = ax, z = ay
    float4          mirror_color;
    float4          mirror_param;   //x = scale, y = ior, when IOR >= 0, we also transmit (just clear glass)
    float4          emission_color;
    float4          emission_param; //x = scale, y = power
}SurfaceParameter;

typedef struct
{
    SurfaceParameter param;
    
    //future texture for parameter blending

}Material;

typedef struct
{
   SurfaceParameter  param;     //chosen surface

   Frame frame;                 //local frame of reference

   float4 localDirFix;          //incoming (fixed) incoming direction, in local
   float4 localGeomN;           //geometry normal (without normal shading)
   
   int materialID;              //material id (Check if necessary, if not remove)
   
   ComponentProbabilities probabilities; //!< Sampling probabilities
}Bsdf;

////////////////////////////////////////////////////////////////////////////
// Albedo methods
////////////////////////////////////////////////////////////////////////////

float AlbedoDiffuse(Bsdf bsdf)
{
   float4 color = bsdf.param.diffuse_color * bsdf.param.diffuse_param.x;
   return Luminance(color.xyz);
}

float AlbedoGlossy(Bsdf bsdf)
{
   float4 color = bsdf.param.glossy_color * bsdf.param.glossy_param.x;
   return Luminance(color.xyz);
}

float AlbedoReflect(Bsdf bsdf)
{
   float4 color = bsdf.param.mirror_color * bsdf.param.mirror_param.x;
   return Luminance(color.xyz);
}

float AlbedoRefract(Bsdf bsdf)
{
   return 0;
}

void GetComponentProbabilities(Bsdf bsdf, ComponentProbabilities *probabilities)
{
   float albedoDiffuse = AlbedoDiffuse(bsdf);
   float albedoGlossy  = AlbedoGlossy(bsdf);
   //float albedoReflect = AlbedoReflect(bsdf);
   //float albedoReflect = mReflectCoeff         * AlbedoReflect(aMaterial);
   //float albedoRefract = (1.f - mReflectCoeff) * AlbedoRefract(aMaterial);
   
   float totalAlbedo   = albedoDiffuse + albedoGlossy;// + albedoReflect + albedoRefract;
   
   if(totalAlbedo < 1e-9f)
   {
      probabilities->diffProb  = 0.f;
      probabilities->glossyProb = 0.f;
      //probabilities.reflProb  = 0.f;
      //probabilities.refrProb  = 0.f;
      //mContinuationProb = 0.f;
   }
   else
   {
      probabilities->diffProb   = albedoDiffuse / totalAlbedo;
      probabilities->glossyProb = albedoGlossy  / totalAlbedo;
      //probabilities.reflProb  = albedoReflect / totalAlbedo;
      //probabilities.refrProb  = albedoRefract / totalAlbedo;
      // The continuation probability is max component from reflectance.
      // That way the weight of sample will never rise.
      // Luminance is another very valid option.
      //mContinuationProb =
      //    (aMaterial.mDiffuseReflectance +
      //    aMaterial.mPhongReflectance +
      //    mReflectCoeff * aMaterial.mMirrorReflectance).Max() +
       //   (1.f - mReflectCoeff);

      //mContinuationProb = std::min(1.f, std::max(0.f, mContinuationProb));
   }
}

Bsdf setupBsdf(global Ray* ray, global Intersection* isect, global Material* materials)//, TextureData texdata, State state
{
   Bsdf bsdf;    //just in case the bsdf is invalid
   
   //frame for local surface
   bsdf.frame = get_frame(isect->n);
   
   //set local dir fix for ray incoming
   bsdf.localDirFix = local_coordinate(bsdf.frame, -ray->d);

   //is bsdf valid
   if(fabs(bsdf.localDirFix.z) < EPS_COSINE)
   {
      bsdf.materialID = -1;
      return bsdf;
   }
   
   bsdf.localGeomN = local_coordinate(bsdf.frame, isect->ng);
   
   //set material id
   bsdf.materialID  = isect->mat;
   
   //choose layer parameter
   bsdf.param = materials[bsdf.materialID].param;
   
   //get probabilities for selecting type of brdf (except emitter)
   GetComponentProbabilities(bsdf, &bsdf.probabilities);

   //return
   return bsdf;
}

bool isBsdfEmitter(Bsdf bsdf)
{
   float4 color = bsdf.param.emission_color * bsdf.param.emission_param.x;
   return Luminance(color.xyz);
}

bool isBsdfInvalid(Bsdf bsdf)
{
   return bsdf.materialID < 0;
}

////////////////////////////////////////////////////////////////////////////
// some bsdf formulas methods
////////////////////////////////////////////////////////////////////////////

//https://graphicscompendium.com/gamedev/15-pbr
float Schlick(float cosTheta, float f0)
{
   return  f0 + (1.f - f0) * pow(1.f - cosTheta, 5.f);
}

//https://schuttejoe.github.io/post/ggximportancesamplingpart1/
//f0 is float r,g,b
float3 SchlickColor(float cosTheta, float3 f0)
{
    float exponential = pow(1.f - cosTheta, 5.f);
    return f0 + (1.f - f0) * exponential;
}

/////////////////////////////////////////////////////////////
// Sampling the GGX Distribution of Visible Normals, Eric Heitz
//////////////////////////////////////////////////////////////


//GGX distribution function or D(N)
float GGXD_N(float4 n, float ax, float ay)
{
   float a2x    = ax * ax;
   float a2y    = ay * ay;
   float axay   = ax * ay;
   float x2n    = n.x * n.x;
   float y2n    = n.y * n.y;
   float z2n    = n.z * n.z;

   float m      = x2n/a2x + y2n/a2y + z2n;

   return 1.f/(M_PI * axay * m * m);
}

//GGX shadow function or G1(V)
float GGX_S(float4 v, float ax, float ay)
{
   float a2x    = ax * ax;
   float a2y    = ay * ay;
   float x2v    = v.x * v.x;
   float y2v    = v.y * v.y;
   float z2v    = v.z * v.z;

   float V_V    = (-1.f + sqrt(1 + (a2x * x2v + a2y * y2v)/z2v))/2.f;
   
   return 1.f/(1.f + V_V);
}





float alpha(float4 w, float ax, float ay)
{
    return sqrt(Cos2Phi(w) * ax * ax + Sin2Phi(w) * ay * ay);
}

//trowbridge-reitz distribution lambda
float lambda(float4 w, float ax, float ay)
{
    float absTanTheta = fabs(TanTheta(w));
    float a           = alpha(w, ax, ay);
    float a2tan2theta = (a * absTanTheta) * (a * absTanTheta);
    float lam         = (-1.f + sqrt(1.f + a2tan2theta))/2;
    return select(lam, 0.f, isinf(absTanTheta));
}

//trowbridge-reitz microfacet distribution
float D_H(float4 wh, float ax, float ay)
{
    float tan2Theta = Tan2Theta(wh);
    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    float e         = (Cos2Phi(wh) / (ax * ax) +
                       Sin2Phi(wh) / (ay * ay)) * tan2Theta;
    float d         =  1.f / (M_PI * ax * ay * cos4Theta * (1.f + e) * (1.f + e));
    return select(d, 0.f, isinf(tan2Theta));
}

//geometric attenuation
float G_Atten(float4 w1, float4 w2, float ax, float ay)
{
    return 1.f/(1.f + lambda(w1, ax, ay) + lambda(w2, ax, ay));
}

float G1(float4 w, float ax, float ay)
{
    return 1.f/(1.f + lambda(w, ax, ay));
}

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
float4 sampleGGXVNDF(float4 v, float ax, float ay, float r1, float r2)
{
   // Section 3.2: transforming the view direction to the hemisphere configuration
   float4 Vh       = normalize((float4)(ax * v.x, ay * v.y, v.z, 0));

   // Section 4.1: orthonormal basis (with special case if cross product is zero)
   float lensq     = Vh.x * Vh.x + Vh.y * Vh.y;
   float4 T1       = select((float4)(1, 0, 0, 0), (float4)(-Vh.y, Vh.x, 0, 0) * rsqrt(lensq), (int4)((lensq > 0) <<31)); 
   float4 T2       = cross(Vh, T1);
   
   // Section 4.2: parameterization of the projected area
   float r         = sqrt(r1);
   float phi       = 2.0 * M_PI * r2;
   float t1        = r * cos(phi);
   float t2        = r * sin(phi);
   float s         = 0.5 * (1.0 + Vh.z);
   t2              = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
   
   // Section 4.3: reprojection onto hemisphere
   float4 Nh       = t1*T1 + t2*T2 + sqrt(max(0.f, 1.f - t1*t1 - t2*t2))*Vh;

   // Section 3.4: transforming the normal back to the ellipsoid configuration
   float4 Ne       = normalize((float4)(ax * Nh.x, ay * Nh.y, max(0.f, Nh.z), 0.f));
   return Ne;
}

//distribution of visible normals(VNDF) or Dv(N)
float VNDF(float4 v, float4 n, float ax, float ay)
{
   float4 Z = (float4)(0, 0, 1, 0);

   return G1(v, ax, ay)* max(0.001f, dot(v,n)) * D_H(n, ax, ay)/max(0.001f, v.z);
}

//ni is from sampled ggxvndf above
float pdfGGXVNDF(float4 v, float4 ni, float ax, float ay)
{
    float DvNi = VNDF(v, ni, ax, ay);
    float VNi  = dot(v, ni);
    return DvNi/(4 * VNi);
    //return VNi == 0.0f ? 0.0f : DvNi/(4 * VNi);
}


///////////////////////////
// From above Torrance_Sparrow Brdf = D(h)*G(wo, wi)*Fr(wo)/(4*cos(wo)*cos(wi))
// based on sampling from sampleGGXVNDF
// and pdf from pdfGGXVNDF
///////////////////////////


//reflect as usual for a mirror
//but when sampling using GGX, localize the incoming vector with sampled Ni and reflect
//     i.e. refl(w, ni) for GGX

float4 reflectFromN(float4 v, float4 n)
{
    return -v + 2 * dot(v, n) * n;
}

float4 reflectLocal(float4 v)
{
    //return (float4)(-v.x, -v.y, v.z, 0);
    return reflectFromN(v, (float4)(0, 0, 1, 0));
}

float4 halfVector(float4 v1, float4 v2)
{
    float4 wh = v1 + v2;
    wh = normalize(wh);
    return wh;
}


////////////////////////////////////////////////////////////////////////////
// BSDF evaluation standard methods
////////////////////////////////////////////////////////////////////////////

float4 PdfDiffuseB(
        Bsdf           bsdf,
        float          *aLocalDirGen,
        float          *oDirectPdfW)
{
  
}

void PdfGlossyB(
        Bsdf           bsdf,
        float          *localDirGen,
        float          *directPdfW)
{

}

float4 PdfReflectB(
        Bsdf           bsdf,
        float          *localDirGen,
        float          *directPdfW)
{
  
}

float4 PdfRefractB(
        Bsdf           bsdf,
        float          *localDirGen,
        float          *directPdfW)
{
  
}

float4 SampleDiffuse(
        Bsdf           bsdf,
        float2         sample,
        float4         *localDirGen,
        float          *pdfW)
{
   if(bsdf.localDirFix.z < EPS_COSINE)
     return (float4)(0);
     
   float unweightedPdfW;
   *localDirGen = SampleCosHemisphereW(sample, &unweightedPdfW);
   *pdfW += unweightedPdfW * bsdf.probabilities.diffProb;

   return (float4)(bsdf.param.diffuse_color.xyz * (float)M_1_PI, 1.f);
}

float4 SampleGlossy(
        Bsdf           bsdf,
        float2         sample,
        float4         *localDirGen,
        float          *pdfW)
{
    if(bsdf.localDirFix.z < EPS_COSINE)
      return (float4)(0.f);

    float ax      = bsdf.param.glossy_param.y;
    float ay      = bsdf.param.glossy_param.z;
    
    float4 wh     = sampleGGXVNDF(bsdf.localDirFix, ax, ay, sample.x, sample.y);
    *localDirGen  = reflectFromN(bsdf.localDirFix, wh);

    float4 wi = bsdf.localDirFix;
    float4 wo = *localDirGen;

    float cosThetaI = fabs(bsdf.localDirFix.z);
    float cosThetaO = fabs((*localDirGen).z);

    *pdfW         += bsdf.probabilities.glossyProb * pdfGGXVNDF(wo, wh, ax, ay);   //ggxPdfReflect(alpha_, wo, (float4)(0, 0, 1, 0), wh);

    float fr      = G_Atten(wi, wo, ax, ay) * D_H(wh, ax, ay)/ (4 * cosThetaI * cosThetaO);               //G_Atten(wi, wo, ax, ay)
    
    float whdot =  fabs(dot(wh, wo));

    float3 Fc     =   SchlickColor(whdot, bsdf.param.glossy_color.xyz);

    if(cosThetaI == 0.f || cosThetaO == 0.f)
       Fc = 0;
    if(wh.x == 0 && wh.y == 0 && wh.z == 0)
       Fc = 0;

    return (float4)(Fc * fr, 0);

}

float4 SampleReflect(
        Bsdf           bsdf,
        float2         sample,
        float4         *localDirGen,
        float          *pdfW)
{
  
}

float4 SampleRefract(
        Bsdf           bsdf,
        float2         sample,
        float4         *localDirGen,
        float          *pdfW)
{

}

float4 EvaluateDiffuse(
        Bsdf           bsdf,
        float4         localDirGen,
        float          *directPdfW)
{
   if(bsdf.probabilities.diffProb == 0)
     return (float4)(0);
     
   if(bsdf.localDirFix.z < EPS_COSINE || localDirGen.z < EPS_COSINE)
     return (float4)(0);
     
   if(directPdfW)
     *directPdfW += bsdf.probabilities.diffProb * fmax(0.f, localDirGen.z * M_1_PI);
     
   return (float4)(bsdf.param.diffuse_color.xyz * M_1_PI, 1.f);
}

float4 EvaluateGlossy(
        Bsdf           bsdf,
        float4         localDirGen,
        float          *directPdfW)
{

   if(bsdf.probabilities.glossyProb == 0)
     return (float4)(0);

   if(bsdf.localDirFix.z < EPS_COSINE || localDirGen.z < EPS_COSINE)
     return (float4)(0);
   
    float ax      = bsdf.param.glossy_param.y;
    float ay      = bsdf.param.glossy_param.z;

    float4 wi     = bsdf.localDirFix;
    float4 wo     = localDirGen;

    float4 wh     = halfVector(wi, wo);

    float cosThetaI = fabs(wi.z);
    float cosThetaO = fabs(wo.z);

    *directPdfW   += bsdf.probabilities.glossyProb * pdfGGXVNDF(wo, wh, ax, ay);   //ggxPdfReflect(alpha_, wo, (float4)(0, 0, 1, 0), wh);

    float fr      = G_Atten(wi, wo, ax, ay) * D_H(wh, ax, ay)/ (4 * cosThetaI * cosThetaO);               //G_Atten(wi, wo, ax, ay)
    
    float whdot = fabs(dot(wh, wo));

    float3 Fc     =  SchlickColor(whdot, bsdf.param.glossy_color.xyz);

    if(cosThetaI == 0.f || cosThetaO == 0.f)
       Fc = 0;
    if(wh.x == 0 && wh.y == 0 && wh.z == 0)
       Fc = 0;

    return (float4)(Fc * fr, 0);

}

float4 EvaluateReflect(
        Bsdf           bsdf,
        float4         localDirGen,
        float          *directPdfW)
{
  
}

float4 EvaluateRefract(
        Bsdf           bsdf,
        float4         localDirGen,
        float          *directPdfW)
{

}

float4 EvaluateBsdf(
        Bsdf        bsdf,
        float4      worldDirGen,
        float       *directPdfW,
        float       *cosThetaGen)
{
   //if not initialized with value, it has an issue (is it always in opencl?)
   float4 result = (float4)(0, 0, 0, 1);

   float4 localDirGen = local_coordinate(bsdf.frame, worldDirGen); 

   if(localDirGen.z * bsdf.localDirFix.z < 0)
       return result;

   *cosThetaGen = localDirGen.z;

   result += EvaluateDiffuse(bsdf, localDirGen, directPdfW);
   result += EvaluateGlossy(bsdf, localDirGen, directPdfW);
   
   return result;
}

float4 PdfBsdf(
        Bsdf        bsdf,
        float4      *worldDirGen,
        bool        *evalRevPdf)
{

}

float4 SampleBsdf(
        Bsdf        bsdf,
        float3      sample,
        float4      *worldDirGen,
        float       *pdfW,
        float       *cosThetaGen)
{
   //select which bsdf to sample
   int sampledEvent;
   if(sample.z < bsdf.probabilities.diffProb)
      sampledEvent = kDiffuse;
   else
      sampledEvent = kGlossy;
      

   float4 result = (float4)(0, 0, 0, 1);
   float4 localDirGen = (float4)(0);

   //sample the selected bsdf and evaluate the rest of the bsdfs
   switch(sampledEvent)
   {
      case kDiffuse:
      {
         result += SampleDiffuse(bsdf, sample.xy, &localDirGen, pdfW);

         if(isFloat4Zero(result))
            return (float4)(0);
         result +=  EvaluateGlossy(bsdf, localDirGen, pdfW);

         break;
      }
      case kGlossy:
      {
         result += SampleGlossy(bsdf, sample.xy, &localDirGen, pdfW);
         if(isFloat4Zero(result))
            return (float4)(0);
         result += EvaluateDiffuse(bsdf, localDirGen, pdfW);

         break;
      }
   }
   
   //calculate costheta from generated direction
   *cosThetaGen   = fabs(localDirGen.z);
   if(*cosThetaGen < EPS_COSINE)
     return (float4)(0);

   //transform the generated local direction to world coordinate
   *worldDirGen = world_coordinate(bsdf.frame, localDirGen);

   return result;
}