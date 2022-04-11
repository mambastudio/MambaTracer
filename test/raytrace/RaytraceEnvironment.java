/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package raytrace;

import bitmap.image.BitmapRGBE;
import cl.data.CColor4;
import cl.data.CInt3;
import coordinate.sampling.sat.SAT;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.FloatValue;

/**
 *
 * @author user
 */
public class RaytraceEnvironment {
    private final OpenCLConfiguration configuration;
    
    //env map
    private CMemory<CColor4> crgb4;
    private CMemory<FloatValue> cenvlum;
    private CMemory<FloatValue> cenvlumsat;
    private final CMemory<CInt3> cenvmapSize;
    
    private SAT sat;
    
    public RaytraceEnvironment(OpenCLConfiguration configuration)
    {
        this.configuration  = configuration;
        this.crgb4          = configuration.createBufferF(CColor4.class, 1, READ_ONLY);
        this.cenvlum        = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
        this.cenvlumsat     = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
        this.cenvmapSize    = configuration.createBufferI(CInt3.class, 1, READ_WRITE);        
    }
    
    public void setEnvironmentMap(BitmapRGBE bitmap)
    {        
        sat = new SAT(bitmap.getWidth(), bitmap.getHeight());
        sat.setArray(bitmap.getLuminanceArray());
        
        crgb4 = configuration.createFromF(CColor4.class, bitmap.getFloat4Data(), READ_ONLY);
        cenvlum = configuration.createFromF(FloatValue.class, sat.getFuncArray(), READ_WRITE);
        cenvlumsat = configuration.createFromF(FloatValue.class, sat.getSATArray(), READ_WRITE);  
        cenvmapSize.mapWriteMemory(envmapsize->{
            CInt3 esize = envmapsize.getCL();
            esize.set(bitmap.getWidth(), bitmap.getHeight(), 1);
        });
    }
    
    public CMemory<CInt3> getEnvMapSize()
    {
        return cenvmapSize;
    }
    
    public CMemory<CColor4> getRgbCL()
    {
        return crgb4;
    }
    
    public CMemory<FloatValue> getEnvLumCL()
    {
        return cenvlum;
    }
        
    public CMemory<FloatValue> getEnvLumSATCL()
    {
        return cenvlumsat;
    }
    
    public void setIsPresent(boolean isPresent)
    {
        cenvmapSize.mapWriteMemory(envmapsize->{
            CInt3 esize = envmapsize.getCL();
            esize.set('z', isPresent ? 1 : 0);
        });
    }    
}
