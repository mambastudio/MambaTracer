float CosTheta(float4 w)
{
   return w.z;
}

float Cos2Theta(float4 w)
{
   return w.z * w.z;
}

float AbsCosTheta(float4 w)
{
   return fabs(w.z);
}

float Sin2Theta(float4 w)
{
   return fmax(0.f, 1.f - Cos2Theta(w));
}

float SinTheta(float4 w)
{
   return sqrt(Sin2Theta(w));
}

float TanTheta(float4 w)
{
   return SinTheta(w)/CosTheta(w);
}

float Tan2Theta(float4 w)
{
   return Sin2Theta(w)/Cos2Theta(w);
}

float CosPhi(float4 w)
{
   float sinTheta = SinTheta(w);
   return select(clamp(w.x/ sinTheta, -1.f, 1.f), 1.f, sinTheta == 0.f); //select is always expressed in reverse of a>b ? a : b --> select(b, a, a>b)
}

float SinPhi(float4 w)
{
   float sinTheta = SinTheta(w);
   return select(clamp(w.y/ sinTheta, -1.f, 1.f), 0.f, sinTheta == 0.f); //select is always expressed in reverse of a>b ? a : b --> select(b, a, a>b)
}

float Cos2Phi(float4 w)
{
   return CosPhi(w) * CosPhi(w);
}

float Sin2Phi(float4 w)
{
   return SinPhi(w) * SinPhi(w);
}

float4 NormalizeVector(float4 w)
{
   return w/length(w);
}

float4 CalculateH(float4 wi, float4 wo)
{
   return NormalizeVector(wi + wo);
}

float4 SphericalToVector(float phi, float theta)
{
   return (float4)(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta), 0);
}

float4 SphericalToVector2(float cosphi, float sinphi, float costheta, float sintheta)
{
   return (float4)(sintheta*cosphi, sintheta*sinphi, costheta, 0);
}

float4 CalculateWoFromH(float4 H, float4 wi)
{
   float4 wo = (float4)(0, 0, 0, 0);
   wo.xyz = 2*(dot(wi.xyz, H.xyz) * H.xyz - wi.xyz);
   return wo;
}
