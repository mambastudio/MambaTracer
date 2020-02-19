/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.data.CInt2;
import cl.core.data.CPoint2;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class CState extends ByteStruct {
    public CInt2 seed;
    public CPoint2 dimension; //width and height of frame
    public float frameCount;
    
    public CState()
    {
        seed = new CInt2();
        dimension = new CPoint2();
        frameCount = 0;
    }
    
    public void setSeed(int seed0, int seed1)
    {
        seed.x = seed0;
        seed.y = seed1;
        this.refreshGlobalArray();
    }
    
    public void setDimension(int dimension0, int dimension1)
    {
        dimension.x = dimension0;
        dimension.y = dimension1;
        this.refreshGlobalArray();
    }
        
    public void incrementFrameCount()
    {
        this.frameCount++;
        this.refreshGlobalArray();
    }
}
