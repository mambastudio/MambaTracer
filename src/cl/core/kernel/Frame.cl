float check(float value)
{
  if(isinf(value)||isnan(value)|| value<0.000001f)
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

__kernel void UpdateImage(global float4*       frameAccum,
                          global float*        frameCount,
                          global int*          imageBuffer)
{
    //get thread id
    int id                     = get_global_id( 0 );
    
    global float4* accumAt     = frameAccum + id;
    global int*    rgbAt       = imageBuffer + id;
    
    float4 color               = (float4)((*accumAt).xyz/frameCount[0], 1);
    color.xyz = pow(color.xyz, (float3)(1.f/1.5f));

    *rgbAt                     = getIntARGB(color);
}

int4 getAverageIntARGB()
{

}

__kernel void UpdateImageJitter(global float4*       frameAccum,
                                global float*        frameCount,
                                global int*          imageBuffer,
                                global float*        width,
                                global float*        height)
{
    //get thread id
    int id                     = get_global_id( 0 );
    global int*    rgbAt       = imageBuffer + id;

    //pixel coordinates
    int pix = id % (int)(*width);
    int piy = id / (int)(*width);
    
    int radius = 0;
    int sum = 0;
    int4 tcolor = (int4)(0, 0, 0, 255);
    for (int ix = pix-radius; ix<pix+radius+1; ix++)
       for (int iy = piy-radius; iy<piy+radius+1; iy++)
       {
            //i + j * w
            int x = min((int)(*width)-1, max(0, ix));
            int y = min((int)(*height)-1, max(0, iy));
            int index = x + y * (int)(*width);

            global float4* accumAt     = frameAccum + index;
            float4 color               = (float4)((*accumAt).xyz/frameCount[0], 1);
            color.xyz                  = pow(color.xyz, (float3)(1.f/1.5f));
            int4 intColor              = clamp255(color);
            tcolor.xyz+=intColor.xyz;
            sum++;
       }

    tcolor.xyz/=sum;
    *rgbAt                     = toIntARGB(tcolor);
}

__kernel void averageAccum(global float4* frameAccum,
                           global float*  frameCount,
                           global float4* frameBuffer)
{
    //get thread id
    int id                 = get_global_id( 0 );
    frameBuffer[id]        = (float4)(frameAccum[id].xyz/frameCount[0], 1.f);
}


__kernel void logLuminanceAverage(global float4* frameBuffer,
                                  global float*  totalLogLuminance,
                                  global float*  totalNumber)
{
    //get thread id
    int id                 = get_global_id( 0 );
    float Lw               = Luminance(frameBuffer[id].xyz);
    float logLw            = log(0.01f + Lw);
    if(Lw > 0)
    {
        atomicAdd(totalLogLuminance, Lw);
        atomicAdd(totalNumber, 1);
    }
}

__kernel void updateFrameImage(
    global float4* frameBuffer,
    global int*    imageBuffer,
    global float*  totalLogLuminance,
    global float*  totalNumber)
{
    //get thread id
    int id = get_global_id( 0 );

    //log average luminance of frameBuffer
    float logAverageLuminance = totalLogLuminance[0]/ totalNumber[0];
    logAverageLuminance       = exp(logAverageLuminance);

    //current color and luminance
    global float4* color      = frameBuffer + id;
    float colorLuminance      = Luminance((*color).xyz);

    if(colorLuminance > 0)
    {
      //scaled luminance -> should be 0.5f but log(delta + Lw) is giving negative values
      float scaledLuminance     = 10.18f * colorLuminance/logAverageLuminance;

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
