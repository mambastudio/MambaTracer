/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CInt2;
import cl.data.CInt4;
import cl.data.CPoint3;
import cl.data.CVector3;
import coordinate.generic.AbstractRay;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CRay extends Structure implements AbstractRay<CPoint3, CVector3>
{
    public CPoint3  o;
    public CVector3 d;    
    public CVector3 inv_d;   
    public CInt4 sign;    
    public CInt2 extra;    
    public float tMin;
    public float tMax;
        
    public CRay() 
    {        
        o = new CPoint3();
        d = new CVector3();
        inv_d = new CVector3();
        sign = new CInt4();
        extra = new CInt2();       
        tMin = 0.01f;
        tMax = Float.POSITIVE_INFINITY;   
    }
    
    public final void init()
    {                
        inv_d = new CVector3(1f/d.get(0), 1f/d.get(1), 1f/d.get(2));
        
        sign.set(0, inv_d.get(0) < 0 ? 1 : 0);
        sign.set(1, inv_d.get(1) < 0 ? 1 : 0);
        sign.set(2, inv_d.get(2) < 0 ? 1 : 0);
        
        this.refreshGlobalArray();
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
        o = new CPoint3(ox, oy, oz);
        d = new CVector3(dx, dy, dz).normalize();
        
        tMin = 0.01f;
        tMax = Float.POSITIVE_INFINITY;
        
        init();
    }

    @Override
    public void set(CPoint3 oc, CVector3 dc) 
    {
        o = oc;
        d = dc.normalize();
        
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
    
    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("o ").append(o.get(0)).append(" ").append(o.get(1)).append(" ").append(o.get(2)).append(" ");
        builder.append("d ").append(d.get(0)).append(" ").append(d.get(1)).append(" ").append(d.get(2));
        return builder.toString();
    }

    @Override
    public AbstractRay<CPoint3, CVector3> copy() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
