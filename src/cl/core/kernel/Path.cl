typedef struct
{
    float4 throughput;
    int active;
    BSDF bsdf;

}Path;

bool Path_isActive(global Path* path)
{
    return  path->active;
}

void Path_Deactivate(global Path* path)
{
    path->active = false;
}

kernel void InitPathData(
    global Path* paths
)
{
    //get global id
    int global_id = get_global_id(0);
    //get path
    global Path* path = paths + global_id;
    //initialize path
    path->throughput          = makeFloat4(1, 1, 1, 1);
    path->active              = true;
    path->bsdf.materialID     = -1;
    path->bsdf.frame.mX       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mY       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mZ       = makeFloat4(0, 0, 0, 0);
    path->bsdf.localDirFix    = makeFloat4(0, 0, 0, 0);
}