__kernel void UpdateImage(global float4*       frameAccum,
                          global float*        frameCount,
                          global int*          imageBuffer,
                          global int*          imageSize)
{
    //get thread id
    int id                     = get_global_id( 0 );
    
    if(id < *imageSize)
    {
        global float4* accumAt     = frameAccum + id;
        global int*    rgbAt       = imageBuffer + id;
    
        float4 color               = (float4)((*accumAt).xyz/frameCount[0], 1);
        color.xyz = pow(color.xyz, (float3)(1.f/2.2f));

        *rgbAt                     = getIntARGB(color);
    }
}


__kernel void averageAccum(global float4*  frameAccum,
                           global float*   frameCount,
                           global float4*  frameBuffer,

                           global float*   loglw,
                           global int*     loglwcount,

                           global int*     imageSize)
{
    //global id
    int id                 = get_global_id( 0 );
    
    //initialize
    loglwcount[id] = 0;
    loglw[id]      = 0;
    
    if(id < *imageSize)
    {
        //get log luminance
        frameBuffer[id]        = (float4)(frameAccum[id].xyz/frameCount[0], 1.f);
        float Lw               = Luminance(frameBuffer[id].xyz);
        
        //is there luminance? do something 
        int logpredicate       = 0;
        float logLwValue       = 0;
        
        if(Lw > 0)
        {
            logLwValue = log(0.01f + Lw);
            logpredicate = 1;
        }

        //set necessary variables
        loglwcount[id] = logpredicate;
        loglw[id] = logLwValue;    
    }
}


__kernel void updateFrameImage(global float4* frameBuffer,
                               global int*    imageBuffer,
                               global float*  totalLogLuminance,
                               global int*    totalNumber,
                               global int*    imageSize)
{
    //get thread id
    int id = get_global_id( 0 );

    //log average luminance of frameBuffer
    float logAverageLuminance = totalLogLuminance[0]/ totalNumber[0];
    logAverageLuminance       = exp(logAverageLuminance);

    if(id < *imageSize)
    {
        //current color and luminance
        global float4* color      = frameBuffer + id;
        float colorLuminance      = Luminance((*color).xyz);
    
        if(colorLuminance > 0)
        {
          //scaled luminance -> should be 0.5f but log(delta + Lw) is giving negative values
          float scaledLuminance     = 0.18f * colorLuminance/logAverageLuminance;

          //tonemap - default Reinhard (experiment with other interesting types... http://filmicworlds.com/blog/filmic-tonemapping-operators/)
          float Y             = toneSimpleReinhard(scaledLuminance);
          
          //check if any error
          Y                   = check(Y);
    
          float4 colorXYZ     = convertRGBtoXYZ(*color);
          float4 xyzColor     = xyz(colorXYZ);
          colorXYZ            = xyYtoXYZ(xyzColor, Y);
          *color              = convertXYZtoRGB(colorXYZ);
    
          //gamma
          frameBuffer[id].xyz = pow(frameBuffer[id].xyz, (float3)(1.f/2.2f));
    
    
          //update frame render
          imageBuffer[id] = getIntARGB(frameBuffer[id]);
        }
    }
}