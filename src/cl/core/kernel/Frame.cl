float check(float value)
{
  if(isinf(value)||isnan(value))
      return 0.f;
  else
      return value;
}

bool isError(float value)
{
  if(isinf(value)||isnan(value))
      return true;
  else
      return false;
}

float Luminance(float3 v)
{
    // Luminance
    return 0.2989f * v.x + 0.5866f * v.y + 0.1145f * v.z;
}

__kernel void averageAccum(global float4* frameAccum,
                           global float*  frameCount,
                           global float4* frameBuffer)
{
    //get thread id
    int id                 = get_global_id( 0 );
    float4 color           = (float4)(frameAccum[id].xyz/frameCount[0], 1.f);
    frameBuffer[id]        =  color;
}

__kernel void logLuminance(global float4* frameBuffer,
                           global float*  logLuminanceBuffer)
{
    //get thread id
    int id                 = get_global_id( 0 );
    logLuminanceBuffer[id] = 0.f;
    float Lw               = Luminance(frameBuffer[id].xyz);
    float logLw            = log(1.f + Lw); //log(0.01f + Lw) -> I don't understand how the negative value is handled  
    logLuminanceBuffer[id] = logLw;
}

__kernel void updateFrameImage(
    global float4* frameBuffer,
    global int*    imageBuffer,
    global float*  totalLogLuminance,
    global float*  totalNumber
)
{
    //get thread id
    int id = get_global_id( 0 );
    
    //log average luminance of frameBuffer
    float logAverageLuminance = totalLogLuminance[0]/ totalNumber[0];
    logAverageLuminance       = exp(logAverageLuminance);
    
    //printFloat(totalNumber[0]);

    //current color and luminance
    global float4* color      = frameBuffer + id;
    float colorLuminance      = Luminance((*color).xyz);

    if(colorLuminance > 0)
    {
      //scaled luminance -> should be 0.5f but log(delta + Lw) is giving negative values
      float scaledLuminance     = 1.6f * colorLuminance/logAverageLuminance;

      //tonemap
      float Y             = scaledLuminance / (1.f + scaledLuminance);
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
