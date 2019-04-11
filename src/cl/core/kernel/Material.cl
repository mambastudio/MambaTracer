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
