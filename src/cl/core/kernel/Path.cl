typedef struct
{
    float4 throughput;        //throughput (multiplied by emission)
    float4 hitpoint;          //position of vertex
    int    pathlength;        //path length or number of segment between source and vertex
    int    lastSpecular;
    float  lastPdfW;
    int active;               //is path active
    BSDF bsdf;                //bsdf (stores local information together with incoming direction)

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
    path->lastSpecular        = true;
    path->lastPdfW            = 1;
    path->active              = true;
    path->bsdf.materialID     = -1;
    path->bsdf.frame.mX       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mY       = makeFloat4(0, 0, 0, 0);
    path->bsdf.frame.mZ       = makeFloat4(0, 0, 0, 0);
    path->bsdf.localDirFix    = makeFloat4(0, 0, 0, 0);
}