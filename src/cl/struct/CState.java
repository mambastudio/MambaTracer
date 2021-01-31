/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CInt2;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CState extends Structure {
    public CInt2 seed;    
    public float frameCount;
    
    public CState()
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
  
    public void setFrameCount(float frameCount)
    {
        this.frameCount = frameCount;
        this.refreshGlobalArray();
    }
}
