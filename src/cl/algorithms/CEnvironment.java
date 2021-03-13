/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import bitmap.Color;
import cl.data.CColor4;
import cl.struct.CEnvironmentGrid;
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
public class CEnvironment {
    private final OpenCLConfiguration configuration;
    private final CMemory<CEnvironmentGrid> cenvgrid;
    
    //env map
    private CMemory<CColor4> crgb4;
    private CMemory<FloatValue> cenvlum;
    private CMemory<FloatValue> cenvlumsat;
    
    private SAT sat;
    
    public CEnvironment(OpenCLConfiguration configuration)
    {
        this.configuration  = configuration;
        this.cenvgrid       = configuration.createBufferB(CEnvironmentGrid.class, 1, READ_WRITE);
        this.crgb4          = configuration.createBufferF(CColor4.class, 1, READ_ONLY);
        this.cenvlum        = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
        this.cenvlumsat     = configuration.createBufferF(FloatValue.class, 1, READ_WRITE);
    }
    
    public void setEnvironmentMap(float[] rgb4, int width, int height)
    {
        float[] lum = new float[width * height];
        for(int i = 0; i<lum.length; i++)
        {
            float x = rgb4[i*4 + 0];
            float y = rgb4[i*4 + 1];
            float z = rgb4[i*4 + 2];
            
            Color col = new Color(x, y, z);
            lum[i] = col.luminance();
        }
        sat = new SAT(width, height);
        sat.setArray(lum);
        
        crgb4 = configuration.createFromF(CColor4.class, rgb4, READ_ONLY);
        cenvlum = configuration.createFromF(FloatValue.class, sat.getFuncArray(), READ_WRITE);
        cenvlumsat = configuration.createFromF(FloatValue.class, sat.getSATArray(), READ_WRITE);
        
        cenvgrid.mapWriteMemory(cgrid->{
            CEnvironmentGrid grid = cgrid.getCL();
            grid.setWidth(width);
            grid.setHeight(height);
        });            
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
        return cenvlumsat   ;
    }
    
    public void setIsPresent(boolean isPresent)
    {
        cenvgrid.mapWriteMemory(cgrid->{
            CEnvironmentGrid grid = cgrid.getCL();
            grid.setIsPresent(isPresent);
        });
    }
    
    public CMemory<CEnvironmentGrid> getCEnvGrid()
    {
        return cenvgrid;
    }
    
    public boolean isPresent()
    {
        CEnvironmentGrid cgrid = cenvgrid.getCL();
        return cgrid.isPresent;
    }
}
