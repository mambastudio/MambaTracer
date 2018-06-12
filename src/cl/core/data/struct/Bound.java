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
public class Bound  extends FloatStruct
{
    CPoint3 minimum;
    CPoint3 maximum;

    public Bound()
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
    public void initFromGlobalArray() {
        float[] globalArray = getGlobalArray();
        if(globalArray == null)
            return;
        int globalArrayIndex = getGlobalArrayIndex();

        minimum.x   = globalArray[globalArrayIndex + 0];
        minimum.y   = globalArray[globalArrayIndex + 1];
        minimum.z   = globalArray[globalArrayIndex + 2];
        maximum.x   = globalArray[globalArrayIndex + 4];
        maximum.y   = globalArray[globalArrayIndex + 5];
        maximum.z   = globalArray[globalArrayIndex + 6];
    }

    @Override
    public float[] getArray() {
        return new float[]{minimum.x, minimum.y, minimum.z, 0,
                           maximum.x, maximum.y, maximum.z, 0};
    }

    @Override
    public int getSize() {
        return 8;
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
