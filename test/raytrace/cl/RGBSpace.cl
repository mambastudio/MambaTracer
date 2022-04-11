#define A 0.15f
#define B 0.50f
#define C 0.10f
#define D 0.20f
#define E 0.02f
#define F 0.30f
#define W 11.2f

float Uncharted2(float Y)
{
   return ((Y*(A*Y+C*B)+D*E)/(Y*(A*Y+B)+D*F))-E/F;
}

float toneUncharted(float Y)
{
   float ExposureBias = 2.0f;
   float curr = Uncharted2(ExposureBias*Y);

   float whiteScale = 1.0f/Uncharted2(W);
   return curr*whiteScale;
}

float toneSimpleReinhard(float Y)
{
   return Y / (1.f + Y);
}

float Luminance(float3 v)
{
    // Luminance
    return 0.2989f * v.x + 0.5866f * v.y + 0.1145f * v.z;
}

float4 convertRGBtoXYZ(float4 color)
{
    float4 colorXYZ;
    
    float r = color.x;
    float g = color.y;
    float b = color.z;
    
    colorXYZ.x = 0.5893f * r + 0.1789f * g + 0.1831f * b;
    colorXYZ.y = 0.2904f * r + 0.6051f * g + 0.1045f * b;
    colorXYZ.z = 0.0000f * r + 0.0684f * g + 1.0202f * b;
    
    return colorXYZ;
}

float4 convertXYZtoRGB(float4 colorXYZ)
{
    float4 rgb;
    
    float X = colorXYZ.x;
    float Y = colorXYZ.y;
    float Z = colorXYZ.z;
    
    rgb.x =  1.967f * X - 0.548f * Y - 0.297f * Z;
    rgb.y = -0.955f * X + 1.938f * Y - 0.027f * Z;
    rgb.z =  0.064f * X - 0.130f * Y + 0.982f * Z;
    rgb.w = 1.f;
    
    return rgb;
}

float4 xyz(float4 colorXYZ)
{
    float4 xyzColor;

    float XYZ = colorXYZ.x + colorXYZ.y + colorXYZ.z;
    if (XYZ < 1e-6f)
        return xyzColor;
    float s = 1.f / XYZ;
    xyzColor.x = colorXYZ.x * s;
    xyzColor.y = colorXYZ.y * s;
    xyzColor.z = colorXYZ.z * s;
    
    return xyzColor;
}

float4 xyYtoXYZ(float4 xyzColor, float Y)
{
    float4 colorXYZ;
    
    float x = xyzColor.x;
    float y = xyzColor.y;

    /*
    colorXYZ.x = y < 1e-6f ? 0 : Y*(x/y);
    colorXYZ.y = Y;
    colorXYZ.z = y < 1e-6f ? 0 : (Y * (1 - x - y))/y;
    */

    if(y < 1e-6f)
      colorXYZ.x = 0.f;
    else
      colorXYZ.x = Y*(x/y);

    colorXYZ.y = Y;

    if(y < 1e-6f)
      colorXYZ.z = 0.f;
    else
      colorXYZ.z = (Y * (1 - x - y))/y;

    return colorXYZ;
}


