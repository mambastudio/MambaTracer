/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CRay;
import coordinate.generic.AbstractBound;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;
/**
 *
 * @author user
 */
public class CBoundingBox extends Struct implements AbstractBound<CPoint3, CVector3, CRay, CBoundingBox>
{
    public cl_float4 minimum;
    public cl_float4 maximum;    
    
    public CBoundingBox() 
    {
        minimum.set(0, Float.POSITIVE_INFINITY); minimum.set(1, Float.POSITIVE_INFINITY); minimum.set(2, Float.POSITIVE_INFINITY);
        maximum.set(0, Float.NEGATIVE_INFINITY); maximum.set(1, Float.NEGATIVE_INFINITY); maximum.set(2, Float.NEGATIVE_INFINITY);         
    }
    
    public CBoundingBox(CPoint3 min, CPoint3 max)
    {
        this();
        include(min);
        include(max);   
       
    }
    
    @Override
    public final void include(CPoint3 p) {
        if (p != null) {
            if (p.x < minimum.get(0))
                minimum.set(0, p.x);
            if (p.x > maximum.get(0))
                maximum.set(0, p.x);
            if (p.y < minimum.get(1))
                minimum.set(1, p.y);
            if (p.y > maximum.get(1))
                maximum.set(1, p.y);
            if (p.z < minimum.get(2))
                minimum.set(2, p.z);
            if (p.z > maximum.get(2))
                maximum.set(2, p.z);
        }
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
    public CBoundingBox getInstance() {
        return new CBoundingBox();
    }
   
    @Override
    public final String toString() {
        return String.format("(%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)", minimum.get(0), minimum.get(1), minimum.get(2), maximum.get(0), maximum.get(1), maximum.get(2));
    }     
}
