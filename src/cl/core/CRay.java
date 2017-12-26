/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.generic.AbstractRay;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.CLTypes.cl_int4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CRay extends Struct implements AbstractRay<CPoint3, CVector3>
{
    public cl_float4 o;
    public cl_float4 d;
    
    public cl_float4 inv_d;    
    public float tMin;
    public float tMax;
    
    public cl_int4 sign;
    
    public CRay() 
    {        
        tMin = 0.01f;
        tMax = Float.POSITIVE_INFINITY;        
    }
    
    public final void init()
    {                
        inv_d = new CVector3(1f/d.get(0), 1f/d.get(1), 1f/d.get(2)).getFloatCL4();
        
        sign.set(0, inv_d.get(0) < 0 ? 1 : 0);
        sign.set(1, inv_d.get(1) < 0 ? 1 : 0);
        sign.set(2, inv_d.get(2) < 0 ? 1 : 0);
    }
    
    public int[] dirIsNeg()
    {
        int[] dirIsNeg = {sign.get(0), sign.get(1), sign.get(2)};
        return dirIsNeg;
    }
    
    public final boolean isInside(float t) 
    {
        return (tMin < t) && (t < tMax);
    }
    
    public CVector3 getInvDir()
    {
        return new CVector3(inv_d);
    }
    
    @Override
    public void set(float ox, float oy, float oz, float dx, float dy, float dz) 
    {       
        o = new CPoint3(ox, oy, oz).getFloatCL4();
        d = new CVector3(dx, dy, dz).normalize().getFloatCL4();
        
        tMin = 0.01f;
        tMax = Float.POSITIVE_INFINITY;
        
        init();
    }

    @Override
    public void set(CPoint3 oc, CVector3 dc) 
    {
        o = oc.getFloatCL4();
        d = dc.normalize().getFloatCL4();
        
        tMin = 0.01f;
        tMax = Float.POSITIVE_INFINITY;
        
        init();
    }

    @Override
    public CPoint3 getPoint() {
        CPoint3 dest = new CPoint3();        
        dest.x = o.get(0) + (tMax * d.get(0));
        dest.y = o.get(1) + (tMax * d.get(1));
        dest.z = o.get(2) + (tMax * d.get(2));
        return dest;
    }

    @Override
    public CPoint3 getPoint(float t) {
        CPoint3 dest = new CPoint3();               
        dest.x = o.get(0) + (t * d.get(0));
        dest.y = o.get(1) + (t * d.get(1));
        dest.z = o.get(2) + (t * d.get(2));
        return dest;
    }

    @Override
    public CVector3 getDirection() {
        return new CVector3(d);
    }

    @Override
    public CVector3 getInverseDirection() {
        return new CVector3(inv_d);
    }

    @Override
    public CPoint3 getOrigin() {
        return new CPoint3(o);
    }

    @Override
    public float getMin() {
        return tMin;
    }

    @Override
    public float getMax() {
        return tMax;
    }
    
}
