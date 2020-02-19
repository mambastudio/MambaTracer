/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.CInt2;
import cl.core.data.CPoint2;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class SState extends ByteStruct {
    public CInt2 seed;    
    public float frameCount;
    
    public SState()
    {
        seed = new CInt2();        
        frameCount = 0;
    }
    
    public void setSeed(int seed0, int seed1)
    {
        seed.x = seed0;
        seed.y = seed1;
        this.refreshGlobalArray();
    }
            
    public void incrementFrameCount()
    {
        this.frameCount++;
        this.refreshGlobalArray();
    }
    
    public void setFrameCount(float frameCount)
    {
        this.frameCount = frameCount;
        this.refreshGlobalArray();
    }
}
