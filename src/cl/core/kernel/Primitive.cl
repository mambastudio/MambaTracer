
//(CONSIDER DELETING)
bool mollerTriangleIntersection(Ray* r, float* tuv, float4 p1, float4 p2, float4 p3)
{
    float4 e1, e2, h, s, q;
     float a, f, b1, b2;
     
     e1 = p2 - p1;
     e2 = p3 - p1;
     h  = cross(r->d, e2);
     a  = dot(e1, h);

     if (a > -0.0000001 && a < 0.0000001)
        return false;

     f = 1/a;

     s = r->o - p1; 
     b1 = f * dot(s, h);
     
     if (b1 < 0.0 || b1 > 1.0)
        return false;

     q = cross(s, e1);
     b2 = f * dot(r->d, q);

     if (b2 < 0.0 || b1 + b2 > 1.0)    
        return false;

     float t = f * dot(e2, q);

     // if intersection, update ray and intersection
     if(isInside(*r, t))
     {
         tuv[0] = t;
         tuv[1] = b1;
         tuv[2] = b2;
         
         return true;
     }
     return false;
}

//(CONSIDER DELETING)
bool intersectTriangle(Ray* r, Intersection* isect, TriangleMesh mesh, int primID)
{
     float4 p1 = getP1(mesh, primID);
     float4 p2 = getP2(mesh, primID);
     float4 p3 = getP3(mesh, primID);
     float4 n  = getNormal(p1, p2, p3);
     float tuv[3];
     
     if(mollerTriangleIntersection(r, tuv, p1, p2, p3))
     {
        //set values
        r->tMax = tuv[0];
        isect->p = getPoint(*r, tuv[0]);
        isect->n = n;
        isect->d = r->d;
        isect->id = primID;     
        isect->mat = mesh.faces[primID].mat;
        return true;
     }
     else
     {
        isect->id = -1;
        isect->mat = -1;
     }
     return false;
}



// triangle intersection
bool mollerTriangleIntersectionGlobal(global Ray* r, float* tuv, float4 p1, float4 p2, float4 p3)
{
    float4 e1, e2, h, s, q;
     float a, f, b1, b2;
     
     e1 = p2 - p1;
     e2 = p3 - p1;
     h  = cross(r->d, e2);
     a  = dot(e1, h);

     if (a > -0.0000001 && a < 0.0000001)
        return false;

     f = 1/a;

     s = r->o - p1; 
     b1 = f * dot(s, h);
     
     if (b1 < 0.0 || b1 > 1.0)
        return false;

     q = cross(s, e1);
     b2 = f * dot(r->d, q);

     if (b2 < 0.0 || b1 + b2 > 1.0)    
        return false;

     float t = f * dot(e2, q);

     // if intersection, update ray and intersection
     if(isInside(*r, t))
     {
         tuv[0] = t;
         tuv[1] = b1;
         tuv[2] = b2;
         
         return true;
     }
     return false;
}

//trinagle mesh intersection
bool intersectTriangleGlobal(global Ray* r, global Intersection* isect, TriangleMesh mesh, int primID)
{
   float4 p1 = getP1(mesh, primID);
   float4 p2 = getP2(mesh, primID);
   float4 p3 = getP3(mesh, primID);
   float4 n  = getNormal(p1, p2, p3);
   float tuv[3];
   
   if(mollerTriangleIntersectionGlobal(r, tuv, p1, p2, p3))
   {
      //set values
      r->tMax = tuv[0];
      isect->p = getPoint(*r, tuv[0]);
      isect->n = n;
      isect->d = r->d;
      isect->id = primID;     
      isect->mat = mesh.faces[primID].mat;
      return true;
   }
   else
   {
      isect->id = -1;
      isect->mat = -1;
   }
   return false;
  
}



// ray box intersection test
bool intersectBox(Ray* r, Intersection* isect, Box box)
{
    //x
    float t1 = (box.min.x - r->o.x)*r->inv_d.x;
    float t2 = (box.max.x - r->o.x)*r->inv_d.x;
    
    float tmin = min(t1, t2);
    float tmax = max(t1, t2);
    
    //y
    t1 = (box.min.y - r->o.y)*r->inv_d.y;
    t2 = (box.max.y - r->o.y)*r->inv_d.y;
    
    tmin = max(tmin, min(min(t1, t2), tmax));
    tmax = min(tmax, max(max(t1, t2), tmin));
    
    //z
    t1 = (box.min.z - r->o.z)*r->inv_d.z;
    t2 = (box.max.z - r->o.z)*r->inv_d.z;
    
    tmin = max(tmin, min(min(t1, t2), tmax));
    tmax = min(tmax, max(max(t1, t2), tmin));
    
    bool hit = tmax > max(tmin, 0.0f);
    float t = (tmin > r->tMin) ? tmin : tmax;

    // if intersection, update ray and intersection
    if(hit && isInside(*r, t))
    {
        // calculate normal
        float4 c = (box.min + box.max) * 0.5f;
        float4 p = getPoint(*r, t) - c;
        float4 d = (box.min - box.max) * 0.5f;
        float bias = 1.00001f;

        float4 n;
        n.x = (int)(p.x/fabs(d.x) * bias);
        n.y = (int)(p.y/fabs(d.y) * bias);
        n.z = (int)(p.z/fabs(d.z) * bias); 
        //n = normalize(n); //not working properly, I don't know why

        //set values
        r->tMax = t;
        isect->p = getPoint(*r, t);
        isect->n = n;
        isect->d = r->d;
        //isect->uv;
        //isect->id;
        //isect->pixel;
        return true;
    }

    return false;
}