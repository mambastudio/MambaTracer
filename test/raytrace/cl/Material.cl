
typedef struct
{     
   int4 diffuseTexture;       //x, y-coord, argb, has-texture(if(w >= 0) ? materialID : false)  //actual texture color
   int4 glossyTexture;
   int4 roughnessTexture;
   int4 mirrorTexture;
   int4 parameters;
  
}TextureData;

//typedef struct
//{
//   int4 baseTexture;                               //x, y-coord, argb  //actual texture color
//   int4 opacityTexture;                            //x, y-coord, argb  //opacity texture
//   int  hasOpacity, hasBaseTex, materialID, paramLevel;
//}TextureData;

Bsdf setupBSDFAreaLight(global Material* materials, TriangleMesh mesh, int triangleIndex)
{
   Bsdf bsdf;
   global Material* material = materials + getMaterial(mesh.faces[triangleIndex].mat);
   bsdf.param = material->param;     //first parameter is the only one to have emission.
   return bsdf;
}

bool isSurfaceEmitter(SurfaceParameter param)
{
   float4 color = param.emission_color * param.emission_param.x;
   return Luminance(color.xyz);
}

float4 getQuickSurfaceColor(SurfaceParameter param, float coeff)
{
   if(isSurfaceEmitter(param))
   {
       return param.emission_color;
   }
   else
   {
       return param.diffuse_color * coeff;
   }
}