float4 make_float4(float x, float y, float z, float w)
{
    float4 res;
    res.x = x;
    res.y = y;
    res.z = z;
    res.w = w;
    return res;
}

typedef struct
{
    float4 m0;
    float4 m1;
    float4 m2;
    float4 m3;
} matrix4x4;

typedef struct
{
    matrix4x4 m;
    matrix4x4 mInv;
} transform;

void print_matrix(matrix4x4 m)
{
     printf("%4.8v4f\n", m.m0);
     printf("%4.8v4f\n", m.m1);
     printf("%4.8v4f\n", m.m2);
     printf("%4.8v4f\n", m.m3);
}

matrix4x4 identity_matrix()
{
    matrix4x4 m;
    m.m0 = (float4)(1, 0, 0, 0); 
    m.m1 = (float4)(0, 1, 0, 0);
    m.m2 = (float4)(0, 0, 1, 0);
    m.m3 = (float4)(0, 0, 0, 1);
    return m;
}

transform identity_transform()
{
    transform trans;
    trans.m    = identity_matrix();
    trans.mInv = identity_matrix();
    return trans;
}

void inverse_transform(transform* transf)
{
    matrix4x4 m    = transf->m;
    matrix4x4 mInv = transf->mInv;
    transf->m    = mInv;
    transf->mInv = m;
}

float4 get_column(matrix4x4 m, int i)
{
    if(i == 0)
       return (float4)(m.m0.x, m.m1.x, m.m2.x, m.m3.x);
    else if(i == 1)
       return (float4)(m.m0.y, m.m1.y, m.m2.y, m.m3.y);
    else if(i == 2)
       return (float4)(m.m0.z, m.m1.z, m.m2.z, m.m3.z);
    else if(i == 3)
       return (float4)(m.m0.w, m.m1.w, m.m2.w, m.m3.w);
}

float4 get_row(matrix4x4 m, int i)
{
    if(i == 0)
       return m.m0;
    else if(i == 1)
       return m.m1;
    else if(i == 2)
       return m.m2;
    else if(i == 3)
       return m.m3;
}

matrix4x4 matrix_from_cols(float4 c0, float4 c1, float4 c2, float4 c3)
{
    matrix4x4 m;
    m.m0 = make_float4(c0.x, c1.x, c2.x, c3.x);
    m.m1 = make_float4(c0.y, c1.y, c2.y, c3.y);
    m.m2 = make_float4(c0.z, c1.z, c2.z, c3.z);
    m.m3 = make_float4(c0.w, c1.w, c2.w, c3.w);
    return m;
}

void set_matrix_column(matrix4x4* m, int i, float4 c)
{
    if(i == 0)
    {
         m->m0.x = c.x;  m->m1.x = c.y; m->m2.x = c.z; m->m3.x = c.w;
    }
    if(i == 1)
    {
         m->m0.y = c.x;  m->m1.y = c.y; m->m2.y = c.z; m->m3.y = c.w;
    }
    if(i == 2)
    {
         m->m0.z = c.x;  m->m1.z = c.y; m->m2.z = c.z; m->m3.z = c.w;
    }
    if(i == 3)
    {
         m->m0.w = c.x;  m->m1.w = c.y; m->m2.w = c.z; m->m3.w = c.w;
    }
}

matrix4x4 matrix_from_rows(float4 c0, float4 c1, float4 c2, float4 c3)
{
    matrix4x4 m;
    m.m0 = c0;
    m.m1 = c1;
    m.m2 = c2;
    m.m3 = c3;
    return m;
}

matrix4x4 matrix_mul_matrix(matrix4x4  m1, matrix4x4 m2)
{
    float4 r0, r1, r2, r3;

    float a0, a1, a2, a3;
    a0 = dot(get_row(m1, 0), get_column(m2, 0));
    a1 = dot(get_row(m1, 0), get_column(m2, 1));
    a2 = dot(get_row(m1, 0), get_column(m2, 2));
    a3 = dot(get_row(m1, 0), get_column(m2, 3));
    r0 = (float4)(a0, a1, a2, a3);
    
    a0 = dot(get_row(m1, 1), get_column(m2, 0));
    a1 = dot(get_row(m1, 1), get_column(m2, 1));
    a2 = dot(get_row(m1, 1), get_column(m2, 2));
    a3 = dot(get_row(m1, 1), get_column(m2, 3));
    r1 = (float4)(a0, a1, a2, a3);

    a0 = dot(get_row(m1, 2), get_column(m2, 0));
    a1 = dot(get_row(m1, 2), get_column(m2, 1));
    a2 = dot(get_row(m1, 2), get_column(m2, 2));
    a3 = dot(get_row(m1, 2), get_column(m2, 3));
    r2 = (float4)(a0, a1, a2, a3);

    a0 = dot(get_row(m1, 3), get_column(m2, 0));
    a1 = dot(get_row(m1, 3), get_column(m2, 1));
    a2 = dot(get_row(m1, 3), get_column(m2, 2));
    a3 = dot(get_row(m1, 3), get_column(m2, 3));
    r3 = (float4)(a0, a1, a2, a3);

    return matrix_from_rows(r0, r1, r2, r3);
}

matrix4x4 matrix_transpose(matrix4x4 m)
{
    return matrix_from_cols(m.m0, m.m1, m.m2, m.m3);
}

float4 matrix_mul_vector4(matrix4x4 m, float4 v)
{
    float4 res;
    res.x = dot(m.m0, v);
    res.y = dot(m.m1, v);
    res.z = dot(m.m2, v);
    res.w = dot(m.m3, v);
    return res;
}

float4 matrix_mul_point4(matrix4x4 m, float4 v)
{
    float4 res;
    res.x = dot(m.m0.xyz, v.xyz) + m.m0.w;
    res.y = dot(m.m1.xyz, v.xyz) + m.m1.w;
    res.z = dot(m.m2.xyz, v.xyz) + m.m2.w;
    return res;
}

float4 transform_point4(matrix4x4 m, float4 v)
{
    float4 res;
    res.x = dot(m.m0.xyz, v.xyz) + m.m0.w;
    res.y = dot(m.m1.xyz, v.xyz) + m.m1.w;
    res.z = dot(m.m2.xyz, v.xyz) + m.m2.w;
    res.w = dot(m.m3.xyz, v.xyz) + m.m3.w;
                                                 //printf("%4.8f\n", res.w);
    if(res.w == 1.f)
    {
      return (float4)(res.x, res.y, res.z, 0);
    }
    else
    {
      res.xyz /= res.w;
      return (float4)(res.xyz, 0);
    }
}

transform camera_transform(float4 position, float4 lookat, float4 up)
{
    float4 d, r, u;
    d.xyz = normalize((lookat - position).xyz);
    r.xyz = normalize(cross(d.xyz, up.xyz).xyz);
    u.xyz = cross(r.xyz, d.xyz);
    
    //translate matrices
    matrix4x4 e    = identity_matrix();
    set_matrix_column(&e,    3, (float4)(-position.x, -position.y, -position.z,  1));
    matrix4x4 eInv = identity_matrix();
    set_matrix_column(&eInv, 3, (float4)( position.x,  position.y,  position.z,  1));

    //orientation matrices
    matrix4x4 view_to_world = identity_matrix();
    set_matrix_column(&view_to_world, 0, r);
    set_matrix_column(&view_to_world, 1, u);
    set_matrix_column(&view_to_world, 2, d * -1);
    matrix4x4 world_to_view = matrix_transpose(view_to_world);

    //camera transform
    transform camera;
    camera.m    = matrix_mul_matrix(world_to_view, e);
    camera.mInv = matrix_mul_matrix(eInv, view_to_world);

    return camera;
}
__kernel void camTest(global CameraStruct* camera)
{
    transform ctransform = camera_transform(camera->position, camera->lookat, camera->up);
    float4 r0;
    r0 = transform_point4(ctransform.mInv, r0);
    print_matrix(ctransform.mInv);
    printFloat4(r0);
}
