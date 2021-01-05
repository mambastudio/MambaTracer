#define FLOATMAX  3.402823e+38f
#define FLOATMIN -3.402823e+38f   //kindly change the naming, since it means least positive zero
#define HIT_MARKER 1
#define MISS_MARKER 0
#define EPS_COSINE 1e-6f
#define EPS_RAY    1e-3f
#define M_PI 3.14159265359f
#define M_1_PI 1.f/M_PI

int getMaterial(int data)
{
    return data & 0xFFFF;
}

int getGroup(int data)
{
    return (data >> 16) & 0xFFFF;
}

float4 makeFloat4(float x, float y, float z, float w)
{
    float4 res;
    res.x = x;
    res.y = y;
    res.z = z;
    res.w = w;
    return res;
}

float3 makeFloat3(float x, float y, float z)
{
    float3 res;
    res.x = x;
    res.y = y;
    res.z = z;
    return res;
}

float2 makeFloat2(float x, float y)
{
    float2 res;
    res.x = x;
    res.y = y;
    return res;
}

int2 makeInt2(int x, int y)
{
    int2 res;
    res.x = x;
    res.y = y;
    return res;
}

float getFloat4At(float4 value, int index)
{
   if(index == 0)
      return value.x;
   else if(index == 1)
      return value.y;
   else if(index == 2)
      return value.z;
   else if (index == 3)
      return value.w;
   else
      return NAN;
}

typedef struct
{
    int width;
    int height;
    bool isPresent;
}EnvMapInfo;

// camera info
typedef struct
{
    float4 position;
    float4 lookat;
    float4 up;
    float2 dimension;
    float fov;
}CameraStruct;

//environment sampling
//Adaptive Environment Sampling on CPU and GPU by Asen Atanasov et al, 2018
typedef struct
{
    bool      isPresent;
    
    //int values encourages better sampling
    int       intLightGrid[5000];   //100 * 50
    int       intTileGrid[2560000]; //16 * 32 * lightGrid

    //for temporary values
    float     floatLightGrid[5000];   //100 * 50
    float     floatTileGrid[2560000]; //16 * 32 * lightGrid
  
}EnvironmentGrid;


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
    global float4       const*  normals;
    global Face         const*  faces;
    int                         size;

}TriangleMesh;

// bounding box
typedef struct
{
   float4 minimum;
   float4 maximum;

}BoundingBox;

void printBound(BoundingBox bound)
{
    printf("[%4.8v4f] [%4.8v4f]\n", bound.minimum, bound.maximum);
}

// ray struct
typedef struct
{
   float4 o;
   float4 d;
   float4 inv_d;
   int4 sign;
   int2 extra;
   float tMin;
   float tMax;
}Ray;

// intersection
typedef struct
{        
   float4 p;
   float4 n;
   float4 d;
   float2 uv;
   int mat;
   int id;
   int hit;  
}Intersection;

//soon to implement unions
typedef struct
{
   int faceId;
   
   float4 p;
   float4 d;
  
}Light;

typedef struct
{
   float4 mX;
   float4 mY;
   float4 mZ;
}Frame;

//http://jcgt.org/published/0006/01/01/
Frame get_frame(float4 n)
{
   Frame frame;
   float sign  = copysign(1.0f, n.z);
   float a     = -1.0f / (sign + n.z);
   float b     = n.x * n.y * a;
   frame.mX    = (float4)(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x, 0);
   frame.mY    = (float4)(b, sign + n.y * n.y * a, -n.y, 0);
   frame.mZ    = n;
   return frame;
}

float4 world_coordinate(Frame frame, float4 a)
{
   return frame.mX*a.x + frame.mY*a.y + frame.mZ*a.z;
}

float4 local_coordinate(Frame frame, float4 a)
{
   return (float4)(dot(frame.mX, a), dot(frame.mY, a), dot(frame.mZ, a), 0);
}

void setRayActive(Ray* ray, bool type)
{
   ray->extra.x = type;
}

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

// normal 1 in mesh
float4 getN1(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.normals[face.nx];
}

// normal 2 in mesh
float4 getN2(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.normals[face.ny];
}

// normal 3 in mesh
float4 getN3(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return mesh.normals[face.nz];
}

bool hasNormals(TriangleMesh mesh, int primID)
{
   Face face  = mesh.faces[primID];
   return face.nx > -1; 
}

bool is_equal_float4(float4 c1, float4 c2)
{
    return (c1.x == c2.x) && (c1.y == c2.y) && (c1.z == c2.z) && (c1.w == c2.w);
}

float4 getCenterOfBoundingBox(BoundingBox bound)
{
   float4 dest;
   dest.x = 0.5f * (bound.minimum.x + bound.maximum.x);
   dest.y = 0.5f * (bound.minimum.y + bound.maximum.y);
   dest.z = 0.5f * (bound.minimum.z + bound.maximum.z);
   return dest;
}

BoundingBox getInitBoundingBox()
{
   BoundingBox res;
   res.maximum = (float4)(FLOATMIN,FLOATMIN,FLOATMIN, 0);
   res.minimum = (float4)(FLOATMAX,FLOATMAX,FLOATMAX, 0);
   return res;
}

BoundingBox unionBoundingBox(BoundingBox bound1, BoundingBox bound2)
{
   BoundingBox res;
   res.minimum = min(bound1.minimum, bound2.maximum);
   res.maximum = max(bound1.maximum, bound2.maximum);
   return res;
}

void addToBoundingBox(BoundingBox* bound, float4 point)
{
   bound->minimum = min(bound->minimum, point);
   bound->maximum = max(bound->maximum, point);
}

BoundingBox getBoundingBox(TriangleMesh mesh, int primID)
{  
    BoundingBox bound = getInitBoundingBox();
    addToBoundingBox(&bound, getP1(mesh, primID));
    addToBoundingBox(&bound, getP2(mesh, primID));
    addToBoundingBox(&bound, getP3(mesh, primID));
    return bound;
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

// get index from pixel
int getIndex(float2 pixel, int width, int height)
{
    return (int)(pixel.x + pixel.y * width);
}

int getSphericalGridIndex(int width, int height, float4 d)
{
    int u, v;
    float phi = 0, theta = 0;
    
    //do conversion of vector direction to u v - [0 to env map size] coordinates
    phi = acos(d.y);
    theta = atan2(d.z, d.x);
    u = (0.5f - 0.5f * theta / M_PI) * width;
    v = (phi / M_PI) * height;
    
    return getIndex((float2)(u, v), width, height);
}

void gammaFloat4(float4* color, float gamma)
{
    (*color).xyz = pow((*color).xyz, (float3)(1.f/gamma));
}

float3 make_float3(float x, float y, float z)
{
    float3 res;
    res.x = x;
    res.y = y;
    res.z = z;
    return res;
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

float4 getFloat4ARGB(int argb)
{
   float4 fargb;
   //a
   fargb.w = ((argb >> 24) & 0xFF) / 255.0f;
   //rgb
   fargb.x = ((argb >> 16) & 0xFF) / 255.0f;
   fargb.y = ((argb >> 8) & 0xFF) / 255.0f;
   fargb.z = (argb & 0xFF) / 255.0f;  

   return fargb;
}

bool isFloat3Zero(float3 value)
{
   union
   {
      float elarray[3];
      float3 elvector;
   } element;
   element.elvector = value;
   for (int i = 0; i < 3; i++)
       if(fabs(element.elarray[i]) != 0.f)
          return false;
   return true;
}

bool isFloat4Zero(float4 value)
{
   union
   {
      float elarray[4];
      float4 elvector;
   } element;
   element.elvector = value;
   for (int i = 0; i < 4; i++)
       if(fabs(element.elarray[i]) != 0.f)
          return false;
   return true;
}

bool isFloat4Valid(float4 value)
{
   union
   {
      float elarray[4];
      float4 elvector;
   } element;
   element.elvector = value;
   for (int i = 0; i < 4; i++)
       if(isnan(element.elarray[i]) || isinf(element.elarray[i]))
          return false;
   return true;
}

bool isFloat4AbsValid(float4 value)
{
   union
   {
      float elarray[4];
      float4 elvector;
   } element;
   element.elvector = value;
   for (int i = 0; i < 4; i++)
       if(isnan(element.elarray[i]) || isinf(element.elarray[i]) || getFloat4At(value, i)<0.f)
          return false;
   return true;
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

float distance_between(float4 p1, float4 p2)
{
   float dx = p2.x - p1.x;
   float dy = p2.y - p1.y;
   float dz = p2.z - p1.z;
   
   return sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

// get ray initialized
void initGlobalRay(global Ray* ray, float4 position, float4 direction)
{
   ray->o = position;
   ray->d = direction;
   ray->inv_d = (float4)(1.f/direction.x, 1.f/direction.y, 1.f/direction.z, 0);
   ray->sign.x = ray->inv_d.x < 0 ? 1 : 0;
   ray->sign.y = ray->inv_d.y < 0 ? 1 : 0;
   ray->sign.z = ray->inv_d.z < 0 ? 1 : 0;
   ray->tMin = 0.001f;
   ray->tMax = INFINITY;
}

// get ray initialized
void initGlobalRayT(global Ray* ray, float4 position, float4 direction, float tMax)
{
   ray->o = position;
   ray->d = direction;
   ray->inv_d = (float4)(1.f/direction.x, 1.f/direction.y, 1.f/direction.z, 0);
   ray->sign.x = ray->inv_d.x < 0 ? 1 : 0;
   ray->sign.y = ray->inv_d.y < 0 ? 1 : 0;
   ray->sign.z = ray->inv_d.z < 0 ? 1 : 0;
   ray->tMin = 0.0001f;
   ray->tMax = tMax;
}

// get ray initialized
void initRay(Ray* ray, float4 position, float4 direction)
{
   ray->o = position;
   ray->d = direction;
   ray->inv_d = (float4)(1.f/direction.x, 1.f/direction.y, 1.f/direction.z, 0);
   ray->sign.x = ray->inv_d.x < 0 ? 1 : 0;
   ray->sign.y = ray->inv_d.y < 0 ? 1 : 0;
   ray->sign.z = ray->inv_d.z < 0 ? 1 : 0;
   ray->tMin = 0.0001f;
   ray->tMax = INFINITY;
}

// pinhole camera ray
void getGlobalCameraRay(global Ray* ray, global CameraStruct* camera, float x, float y, float width, float height)
{
     float fv    = radians(camera->fov);

     float4 look = camera->lookat - camera->position;
     float4 Du   = cross(look, camera->up); Du = normalize(Du);
     float4 Dv   = cross(look, Du);        Dv = normalize(Dv);
     
     float fl    = width / (2.f * tan(0.5f * fv));
     float4 vp   = normalize(look);

     vp = vp*fl - 0.5f*(width*Du + height*Dv);

     float4 dir  =  x*Du + y*Dv + vp; dir = normalize(dir);
     
     initGlobalRay(ray, camera->position, dir)  ;
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

// ray bounding box intersection
bool intersectBoundT(Ray r, BoundingBox bound, float* t)
{   
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
    t[0] = tmin;
    t[1] = tmax;
    return ((tmin < r.tMax) && (tmax > r.tMin));
   //float tmax
}

//https://stackoverflow.com/questions/29866889/save-a-float-into-an-integer-without-losing-floating-point-precision
int castFloatToInt(float f)
{
    union { float f; int i; } u;
    u.f = f;
    return u.i;
}


/*
     OpenCL 1.2 doesn't have atomic operations on floats...

     -details well on implementation, and accomodates other operations
        *https://stackoverflow.com/questions/18950732/atomic-max-for-floats-in-opencl
     -simple but only for max and min, which I've adopted here
        *https://ingowald.blog/2018/06/24/float-atomics-in-opencl/
*/

//Function to perform the atomic max
inline void atomicMax(volatile __global float *source, float operand) {
    union {
           unsigned int u32;
           float        f32;
    } next, expected, current;
    current.f32    = *source;
    do {
   	   expected.f32 = current.f32;
           next.f32     = max(expected.f32, operand);
           current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)source,
                               expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

//Function to perform the atomic min
inline void atomicMin(volatile __global float *source, float operand) {
   union {
           unsigned int u32;
           float        f32;
    } next, expected, current;
    current.f32    = *source;
    do {
   	   expected.f32 = current.f32;
           next.f32     = min(expected.f32, operand);
           current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)source,
                               expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

//Function to perform the atomic add
inline void atomicAdd(volatile __global float *source, float operand) {
   union {
           unsigned int u32;
           float        f32;
    } next, expected, current;
    current.f32    = *source;
    do {
   	   expected.f32 = current.f32;
           next.f32     = expected.f32 + operand;
           current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)source,
                               expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

//Function to perform the atomic mul
inline void atomicMul(volatile __global float *source, float operand) {
   union {
           unsigned int u32;
           float        f32;
    } next, expected, current;
    current.f32    = *source;
    do {
   	   expected.f32 = current.f32;
           next.f32     = expected.f32 * operand;
           current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)source,
                               expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

void atomicAddFloat4(volatile __global float4* ptr, float4 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomicAdd(p    , value.x);
    atomicAdd(p + 1, value.y);
    atomicAdd(p + 2, value.z);
    atomicAdd(p + 3, value.w);
}

void atomicMulFloat4(volatile __global float4* ptr, float4 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomicMul(p    , value.x);
    atomicMul(p + 1, value.y);
    atomicMul(p + 2, value.z);
    atomicMul(p + 3, value.w);
}

__kernel void findBound(
    //group to look for instances
    global const int* groupIndex,

     //mesh
    global const float4* points,
    global const float4* normals,
    global const Face*   faces,
    global const int*    size,

    //global bound of size 6 -> xmin, ymin, zmin, xmax, ymax, zmax
    global const float* groupBound
)
{
    //global id  for mesh at index (id)
    int id= get_global_id( 0 );

    //Scene mesh
    TriangleMesh mesh = {points, normals, faces, size[0]};

    //Get face at id
    global Face * face = faces + id;
    
    //Get bound coordinates
    global float* xmin = groupBound + 0;
    global float* ymin = groupBound + 1;
    global float* zmin = groupBound + 2;
    global float* xmax = groupBound + 3;
    global float* ymax = groupBound + 4;
    global float* zmax = groupBound + 5;


    //update bounds
    int groupID = getMaterial(face-> mat);
    //printlnInt(groupID);
    if(groupIndex[0] == groupID)
    {
        //printlnInt(*groupIndex);
        BoundingBox bounds = getBoundingBox(mesh, id);

        //bound->minimum = min(bound->minimum, point);
        atomicMin(xmin, bounds.minimum.x);
        atomicMin(ymin, bounds.minimum.y);
        atomicMin(zmin, bounds.minimum.z);

        //bound->maximum = max(bound->maximum, point);
        atomicMax(xmax, bounds.maximum.x);
        atomicMax(ymax, bounds.maximum.y);
        atomicMax(zmax, bounds.maximum.z);
        
        BoundingBox box;
        box.minimum = (float4)(*xmin, *ymin, *zmin, 0);
        box.maximum = (float4)(*xmax, *ymax, *zmax, 0);
        //printBound(box);
    }
}

void InitIsect(global Intersection* isect)
{
    isect->p             = (float4)(0, 0, 0, 0);
    isect->n             = (float4)(0, 0, 0, 0);
    isect->d             = (float4)(0, 0, 0, 0);
    isect->uv            = (float2)(0, 0);
    isect->id            = -1;
    isect->hit           = MISS_MARKER;
    isect->mat           = -1;
}

__kernel void InitIntersection(global Intersection* isects)
{
    int global_id = get_global_id(0);
    global Intersection* isect = isects + global_id;
    
    InitIsect(isect);
}

__kernel void InitIntDataToIndex(global int* intData)
{
    int id= get_global_id( 0 );   
    intData[id] = id; 
}

__kernel void InitIntData(
    global int* array)
{
    uint global_id = get_global_id(0);
    array[global_id] = 0;
}


__kernel void InitFloat4DataXYZ(global float4* float4Data)
{
    int id = get_global_id(0);
    float4Data[id] = (float4)(0, 0, 0, 1);
}