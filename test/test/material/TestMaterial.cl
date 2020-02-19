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


kernel void test(global Material* materials)
{
   //get thread id
   int id = get_global_id( 0 );
   
   global Material* material = materials + id;


   material-> iorEnabled     = true;
   material->emitter         = (float4)(1, 1, 31, 1);
   material->eu              = 34;
}
