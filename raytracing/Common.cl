// camera info
typedef struct
{
    float4 position;
    float4 lookat;
    float4 up;
    float fov;
}CameraStruct;

// box
typedef struct
{
   float4 min;
   float4 max;
}Box;

// triangle face indices
typedef struct
{
    int vx,  vy,  vz;
    int uvx, uvy, uvz;
    int nx,  ny,  nz;
    int mat;
  
}Face;

// triangle mesh
typedef struct
{
    global float4       const*  points;
    global Face         const*  faces;
    int                         size;

}TriangleMesh;

// bounding box
typedef struct
{
   float4 minimum;
   float4 maximum;
}BoundingBox;


// ray struct
typedef struct
{
   float4 o;
   float4 d;

   float4 inv_d;
   float4 oxinv_d;
   float tMin;
   float tMax;
   
   int4 sign;
}Ray;

// intersection
typedef struct
{
   float4 p;
   float4 n;
   float2 uv;
   int id;
   float2 pixel;
}Intersection;

// vertex 1 in mesh
float4 getP1(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.points[face.vx];
}

// vertex 2 in mesh
float4 getP2(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.points[face.vy];
}

// vertex 3 in mesh
float4 getP3(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.points[face.vz];
}

float4 getNormal(float4 p1, float4 p2, float4 p3)
{
   float4 e1 = p2 - p1;
   float4 e2 = p3 - p1;
   float4 n  = cross(e1, e2);
   n = normalize(n);
   
   return n;
}

void dirNegArray(Ray r, int* isDirNegative)
{
   isDirNegative[0] = r.sign.x;
   isDirNegative[1] = r.sign.y;
   isDirNegative[2] = r.sign.z;
}

// get pixel coordinate from index
float2 getPixel(int index, int width, int height)
{
    float2 pixel;
    pixel.x = index % width;
    pixel.y = index / width;
    return pixel;
}

//int rgb from float3
int getIntRGB(float3 color)
{
   int rgb;

   int r = (int)(clamp( color.x, 0.f, 1.f ) * 255.0f);
   int g = (int)(clamp( color.y, 0.f, 1.f ) * 255.0f);
   int b = (int)(clamp( color.z, 0.f, 1.f ) * 255.0f);
   rgb = (r << 16)|(g << 8)|b;
   
   return rgb;
}

//int rgb from float3
int getIntARGB(float4 color)
{
   int rgb;

   int r = (int)(clamp( color.x, 0.f, 1.f ) * 255.0f);
   int g = (int)(clamp( color.y, 0.f, 1.f ) * 255.0f);
   int b = (int)(clamp( color.z, 0.f, 1.f ) * 255.0f);
   int a = (int)(clamp( color.w, 0.f, 1.f ) * 255.0f);
   rgb = (a << 24)|(r << 16)|(g << 8)|b;
   
   return rgb;
}

// is distance 't' within ray boundary
bool isInside(Ray r, float t)
{
   return (r.tMin < t) && (t < r.tMax);
}

// get point from t
float4 getPoint(Ray r, float t)
{
   float4 point;
   point.x = r.o.x + t * r.d.x;
   point.y = r.o.y + t * r.d.y;
   point.z = r.o.z + t * r.d.z;
   return point;
}

// get ray initialized
Ray initRay(float4 position, float4 direction)
{
   Ray ray;
   
   ray.o = position;
   ray.d = direction;
   ray.inv_d = (float4)(1.f/direction.x, 1.f/direction.y, 1.f/direction.z, 0);
   ray.oxinv_d = -ray.o * ray.inv_d;
   ray.sign.x = ray.inv_d.x < 0 ? 1 : 0;
   ray.sign.y = ray.inv_d.y < 0 ? 1 : 0;
   ray.sign.z = ray.inv_d.z < 0 ? 1 : 0;
   ray.tMin = 0.001f;
   ray.tMax = INFINITY;

   return ray;
}

// pinhole camera ray
Ray getCameraRay(float x, float y, float width, float height, CameraStruct camera)
{
     float fv    = radians(camera.fov);

     float4 look = camera.lookat - camera.position;
     float4 Du   = cross(look, camera.up); Du = normalize(Du);
     float4 Dv   = cross(look, Du);        Dv = normalize(Dv);
     
     float fl    = width / (2. * tan(0.5f * fv));
     float4 vp   = normalize(look);

     vp = vp*fl - 0.5f*(width*Du + height*Dv);

     float4 dir  =  x*Du + y*Dv + vp; dir = normalize(dir);
     
     return initRay(camera.position, dir)  ;
}

// get value from index
float get(float4 value, int index)
{
        if(index == 0) return value.x;
   else if(index == 1) return value.y;
   else if(index == 2) return value.z;
   else                return -1;
}

// set value in prescribed index
void set(float4 coordinate, float value, int index)
{
        if(index == 0) coordinate.x = value;
   else if(index == 1) coordinate.y = value;
   else if(index == 2) coordinate.z = value;
}

// get value from index
int getInt(int4 value, int index)
{
        if(index == 0) return value.x;
   else if(index == 1) return value.y;
   else if(index == 2) return value.z;
   else                return -1;
}

float4 getExtent(int index, BoundingBox bound)
{
   float4 value;
        if(index == 0)      value = bound.minimum;
   else if(index == 1)      value = bound.maximum;
   return value;
}


float min3(float a, float b, float c)
{
    return min(min(a, b), c);
}

float max3(float a, float b, float c)
{
    return max(max(a, b), c);
}

// ray bounding box intersection
bool intersectBound(Ray r, BoundingBox bound)
{
   //float4 const f = mad(bound.maximum, r.inv_d, r.oxinv_d);
   //float4 const n = mad(bound.minimum, r.inv_d, r.oxinv_d);
   //float4 const tmax = max(f, n);
   //float4 const tmin = min(f, n);
   //float const t1 = min(min3(tmax.x, tmax.y, tmax.z), r.tMax);
   //float const t0 = max(max3(tmin.x, tmin.y, tmin.z), 0.f);

   //return t0<=t1;
   
    float tmin  = (getExtent(r.sign.x, bound).x - r.o.x) * r.inv_d.x;
    float tmax  = (getExtent(1-r.sign.x, bound).x - r.o.x) * r.inv_d.x;
    float tymin = (getExtent(r.sign.y, bound).y - r.o.y) * r.inv_d.y;
    float tymax = (getExtent(1-r.sign.y, bound).y - r.o.y) * r.inv_d.y;
    if ( (tmin > tymax) || (tymin > tmax) )
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    float tzmin  = (getExtent(r.sign.z, bound).z - r.o.z) * r.inv_d.z;
    float tzmax  = (getExtent(1-r.sign.z, bound).z - r.o.z) * r.inv_d.z;
    if ( (tmin > tzmax) || (tzmin > tmax) )
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    return ((tmin < r.tMax) && (tmax > r.tMin));
   //float tmax
}

// print float value
void printFloat(float v)
{
    printf("%4.8f\n", v);
}

void printFloat3(float3 v)
{
    printf("%4.8v3f\n", v);
}

// print float4
void printFloat4(float4 v)
{
    printf("%4.8v4f\n", v);
}

// print int
void printInt(int i, bool newLine)
{
    if(newLine)
        printf("%2d\n", i);
    else
        printf("%2d  ", i);
}

// print boolean
void printBoolean(bool value)
{
    printf(value ? "true \n" : "false \n");
}