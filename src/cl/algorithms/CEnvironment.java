/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import bitmap.image.BitmapRGBE;
import cl.data.CColor4;
import cl.data.CInt3;
import cl.data.CPoint3;
import cl.struct.CLightGrid;
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
public final class CEnvironment {
    private final OpenCLConfiguration configuration;
    private final CAdaptiveEnvironment adaptiveEnv;
    
    //env map
    private CMemory<CColor4> crgb4;
    private CMemory<FloatValue> cenvlum;
    private CMemory<FloatValue> cenvlumsat;
    private final CMemory<CInt3> cenvmapSize;
    
    private SAT sat;
    
    public CEnvironment(OpenCLConfiguration configuration)
    {
        this.configuration  = configuration;
        this.crgb4          = configuration.createBufferF(CColor4.class, 1, READ_ONLY);
        this.cenvlum        = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
        this.cenvlumsat     = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
        this.cenvmapSize    = configuration.createBufferI(CInt3.class, 1, READ_WRITE);
        this.adaptiveEnv    = new CAdaptiveEnvironment(configuration);
    }
    
    public void setEnvironmentMap(BitmapRGBE bitmap)
    {        
        sat = new SAT(bitmap.getWidth(), bitmap.getHeight());
        sat.setArray(bitmap.getLuminanceArray());
        
        adaptiveEnv.setHDRLuminance(sat);
        
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
    
    public void setCameraPosition(CPoint3 cameraPosition)
    {
        adaptiveEnv.setCameraPosition(cameraPosition);
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
        adaptiveEnv.setIsPresent(isPresent);
        cenvmapSize.mapWriteMemory(envmapsize->{
            CInt3 esize = envmapsize.getCL();
            esize.set('z', isPresent ? 1 : 0);
        });
    }
    
    public CMemory<CLightGrid> getLightGrid()
    {
        return adaptiveEnv.getLightGridCL();
    }
    
    public boolean isPresent()
    {
        return adaptiveEnv.isPresent();
    }
    
    public void resetAdaptive()
    {
        adaptiveEnv.resetHDRLuminance();
    }
    
    public void adaptiveUpdate()
    {
        adaptiveEnv.update();
    }
}
