__kernel void identifyMeshLights(
    //mesh faces    
    global const Face*      faces,
    global const int*       size,
        
    global Material*        materials,
    global int*             predicate
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    if(id < *size)
    {
        //Get face at id
        global Face * face = faces + id;

        //get material
        int matID = getMaterial(face-> mat);
        global Material* material = materials + matID;
        
        SurfaceParameter param = material->param1;
        if(param.brdfType == EMITTER)
           predicate[id] = 1;
        else
           predicate[id] = 0;

    }
}

__kernel void  prepareLightInfo(
                __global int*          predicate,
                __global int*          prefixsum,
                __global int*          totalSize,
                //light info and count
                __global LightInfo*    lightInfoList
               )
{
    //get thread id
    int global_id = get_global_id( 0 );

    if(global_id < *totalSize)
    {
      if(predicate[global_id])
      {
           int lIndex = prefixsum[global_id];

           global LightInfo* lightInfo = lightInfoList + lIndex;

           lightInfo->faceId = global_id;
           lightInfo->type = AREA_LIGHT;

      }
    }
}
