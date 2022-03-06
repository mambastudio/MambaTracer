typedef struct
{
    float4 throughput;        //throughput (multiplied by emission)
    float4 hitpoint;             //position of vertex
    int    pathlength;           //path length or number of segment between source and vertex
    int    lastSpecular;
    float  lastPdfW;
    int    active;               //is path active
    Bsdf   bsdf;                 //bsdf (stores local information together with incoming direction)

}Path;
