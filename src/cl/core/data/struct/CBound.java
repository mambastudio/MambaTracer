/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.CBoundingBox;
import cl.core.data.CPoint3;
import coordinate.struct.FloatStruct;

/**
 *
 * @author user
 */
public class CBound  extends FloatStruct
{
    public CPoint3 minimum;
    public CPoint3 maximum;

    public CBound()
    {
        this.minimum = new CPoint3();
        this.maximum = new CPoint3();
    }
    
    public void setBound(CBoundingBox bound)
    {
        minimum = bound.getMinimum();
        maximum = bound.getMaximum();
    }
    
    public CBoundingBox getCBound()
    {
        CBoundingBox cbound = new CBoundingBox();
        cbound.include(minimum);
        cbound.include(maximum);
        return cbound;
    }
    
    public CPoint3 getMinimum()
    {
        return minimum;
    }
    
    public CPoint3 getMaximum()
    {
        return maximum;
    }

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder(); 
        builder.append("Bounding Box").append("\n");
        builder.append("  ").append("minimum: ").append(minimum).append("\n");
        builder.append("  ").append("maximum: ").append(maximum).append("\n");
        return builder.toString();
    }
}   
