#define NODEROOT() (mesh.size)

/*
  while(true)
  {
     generateCameraRays(camera, rays, isects, width, height);
     intersectScene(rays, isects, atomic_count, 
                    points, faces, size, 
                    nodes, bounds);
     updateShadeImage(imageBuffer, width, height, isects);
  }
*/
__kernel void fastShade(
    global Material* materials,
    global Intersection* isects,
    global int* imageBuffer
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //default color
    float4 color = (float4)(0, 0, 0, 1);
    float4 color1 = (float4)(1, 1, 1, 1);

    //get intersection and material
    global Intersection* isect = isects + id;
    
    if(isect->hit)
    {
        float coeff = fabs(dot(isect->d, isect->n));
        color.xyz   = getMaterialColor(materials[isect->mat], coeff).xyz;
        imageBuffer[id] = getIntARGB(color);
    }
}

__kernel void backgroundShade(
    global Intersection* isects,
    global CameraStruct* camera,
    global int* imageBuffer
)
{
    //get thread id
    int id = get_global_id( 0 );

    //updated the intersected areas color
    global Intersection* isect = isects + id;
    if(!isect->hit)
    {
        //update
        imageBuffer[id] = getIntARGB((float4)(0, 0, 0, 1));
    }
}

__kernel void updateNormalShadeImage(
    global Intersection* isects,
    global CameraStruct* camera,
    global int* imageBuffer
)
{
    int id= get_global_id( 0 );

    //updated the intersected areas color
    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        //shade normal if facing camera or not
        float ndotd = dot(isect->d, isect->nDefault);
        float4 shade = ndotd < 0 ? (float4)(1, 0, 0, 1) : (float4)(0, 0, 1, 1);
        shade.xyz *= fabs(ndotd);

        //update
       imageBuffer[id] = getIntARGB(shade);
    }
}


__kernel void updateGroupbufferShadeImage(
    global Intersection* isects,
    global CameraStruct* camera,
    global int* groupBuffer
)
{
    int id= get_global_id( 0 );

    global Intersection* isect = isects + id;
    if(isect->hit)
    {
        groupBuffer[id] = getMaterial(isect->mat);
    }  
}