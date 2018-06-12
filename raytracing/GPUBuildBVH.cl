typedef struct
{
   BoundingBox box;
   float4 data;
}Node;

__kernel void calculateMorton(global float4* points, global Face* faces, global int* size, global Node* nodes, global int2* mortonPrimitive)
{
   int id= get_global_id( 0 );
   TriangleMesh mesh = {points, faces, size[0]};
}
