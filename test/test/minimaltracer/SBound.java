/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.generic.AbstractBound;
import coordinate.struct.FloatStruct;

/**
 *
 * @author user
 */
public class SBound extends FloatStruct implements AbstractBound<CPoint3, CVector3, SRay, SBound>
{
    public CPoint3 minimum;
    public CPoint3 maximum;    
    
    public SBound() 
    {
        minimum = new CPoint3();
        maximum = new CPoint3();
        
        minimum.set('x', Float.POSITIVE_INFINITY); 
        minimum.set('y', Float.POSITIVE_INFINITY); 
        minimum.set('z', Float.POSITIVE_INFINITY);
        
        maximum.set('x', Float.NEGATIVE_INFINITY); 
        maximum.set('y', Float.NEGATIVE_INFINITY); 
        maximum.set('z', Float.NEGATIVE_INFINITY);         
    }
    
    public SBound(CPoint3 min, CPoint3 max)
    {
        this();
        include(min);
        include(max);   
       
    }
    
    public void setBound(SBound bound)
    {
        minimum = bound.getMinimum();
        maximum = bound.getMaximum();
        this.refreshGlobalArray();
    }
    
    public SBound getCBound()
    {
        SBound cbound = new SBound();
        cbound.include(minimum);
        cbound.include(maximum);
        return cbound;
    }
    
    @Override
    public final void include(CPoint3 p) {
        if (p != null) {
            if (p.x < minimum.get(0))
                minimum.set('x', p.x);
            if (p.x > maximum.get(0))
                maximum.set('x', p.x);
            if (p.y < minimum.get(1))
                minimum.set('y', p.y);
            if (p.y > maximum.get(1))
                maximum.set('y', p.y);
            if (p.z < minimum.get(2))
                minimum.set('z', p.z);
            if (p.z > maximum.get(2))
                maximum.set('z', p.z);
        }
        this.refreshGlobalArray();
    }

    @Override
    public CPoint3 getCenter() {
        CPoint3 dest = new CPoint3();
        dest.x = 0.5f * (minimum.get(0) + maximum.get(0));
        dest.y = 0.5f * (minimum.get(1) + maximum.get(1));
        dest.z = 0.5f * (minimum.get(2) + maximum.get(2));
        return dest;
    }

    @Override
    public float getCenter(int dim) {
        return getCenter().get(dim);
    }

    @Override
    public CPoint3 getMinimum() {
        return new CPoint3(minimum);
    }

    @Override
    public CPoint3 getMaximum() {
        return new CPoint3(maximum);
    }

    @Override
    public SBound getInstance() {
        return new SBound();
    }
   
    @Override
    public final String toString() {
        return String.format("(%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)", minimum.get(0), minimum.get(1), minimum.get(2), maximum.get(0), maximum.get(1), maximum.get(2));
    }     
}
