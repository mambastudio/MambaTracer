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
   bsdf.localDirFix = local_coordinate(bsdf.frame, -ray->d);

   return bsdf;
}

float4 getMaterialColor(Material mat, float coeff)
{
   if(mat.emitterEnabled) return mat.emitter;
   else                   return mat.diffuse * coeff;
}

float4 getEmitterColor(Material mat)
{
   return mat.emitter * 45.f;
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