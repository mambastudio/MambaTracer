typedef struct
{
   float4 diffuse;         //diffuse    - r, g, b, w (pad)
   float diffuseWeight;     //diffuse    - diffuse weight

   float4 reflection;      //reflection - r, g, b, w (pad)
   float eu, ev, ior;       //reflection - eu, ev, ior
   bool iorEnabled;      //reflection - transmission enabled

   float4 emitter;         //emission   - r, g, b, w (power)
   bool emitterEnabled;  //emission   - emission enabled
}Material;

float4 getMaterialColor(Material mat, float coeff)
{
   if(mat.emitterEnabled) return mat.emitter;
   else                   return mat.diffuse * coeff;
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