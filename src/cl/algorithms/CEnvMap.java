/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.core.data.CColor4;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CEnvMap {
    private final OpenCLConfiguration configuration;
    private CMemory<CColor4> crgb4;
    private final CMemory<IntValue> cenvWidth;
    private final CMemory<IntValue> cenvHeight;
    private final CMemory<IntValue> cisEnvPresent;
    
    public CEnvMap(OpenCLConfiguration configuration)
    {
        this.configuration  = configuration;
        this.crgb4          = configuration.createBufferF(CColor4.class, 1, READ_ONLY);
        this.cenvWidth      = configuration.createValueI(IntValue.class, new IntValue(1), READ_ONLY);
        this.cenvHeight     = configuration.createValueI(IntValue.class, new IntValue(1), READ_ONLY);
        this.cisEnvPresent  = configuration.createValueI(IntValue.class, new IntValue(0), READ_ONLY);
    }
    
    public void setEnvironmentMap(float[] rgb4, int width, int height)
    {
        crgb4 = configuration.createFromF(CColor4.class, rgb4, READ_ONLY);
        cenvWidth.setCL(new IntValue(width));
        cenvHeight.setCL(new IntValue(height));        
    }
    
    public CMemory<CColor4> getRgbCL()
    {
        return crgb4;
    }
    
    public CMemory<IntValue> getWidthCL()
    {
        return cenvWidth;
    }
    
    public CMemory<IntValue> getHeightCL()
    {
        return cenvHeight;
    }
    
    public CMemory<IntValue> getIsPresentCL()
    {
        return cisEnvPresent;
    }
    
    
    public void setIsPresent(boolean isPresent)
    {
        if(isPresent)
            cisEnvPresent.setCL(new IntValue(1));
        else
            cisEnvPresent.setCL(new IntValue(0));
    }
}
