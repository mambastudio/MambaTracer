/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.struct.CIntersection;
import cl.main.TracerAPI;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class CCompact {
    TracerAPI api;
    
    private final PrefixSum prefixsum;
    
    private CStructTypeBuffer<CIntersection>    isects;
    private CStructTypeBuffer<CIntersection>    temp_isects;
    private CKernel                             initTempIsectsKernel;
    private CKernel                             compactIsectsKernel;
    private CKernel                             transferIsectsKernel;
    
    private CIntBuffer                          pixels;
    private CIntBuffer                          temp_pixels;
    private CKernel                             initTempPixelsKernel;
    private CKernel                             compactPixelsKernel;
    private CKernel                             transferPixelsKernel;
    
    public CCompact(TracerAPI api)
    {        
        this.api = api;        
        this.prefixsum = new PrefixSum(api.configurationCL());
    }
    
    public void initPrefixSumFrom(CStructTypeBuffer<CIntersection> isectBuffer, CIntBuffer total)
    {
        prefixsum.init(isectBuffer, total);
    }
    
    public void initPixels(CIntBuffer pixels)
    {
        this.pixels = pixels;
        this.temp_pixels = api.allocInt("temp_pixels", pixels.getBufferSize(), READ_WRITE);
        this.initTempPixelsKernel = api.createKernel("InitIntData", temp_pixels);
        this.compactPixelsKernel = api.createKernel("compactPixels", this.pixels, temp_pixels, prefixsum.getPredicate(), prefixsum.getPrefixSum());
        this.transferPixelsKernel = api.createKernel("transferPixels", this.pixels, temp_pixels);
    }
    
    public void initIsects(CStructTypeBuffer<CIntersection> isects)
    {
        
        this.isects = isects;
        this.temp_isects = api.allocStructType("temp_isects", CIntersection.class, isects.getSize(), READ_WRITE); 
        this.initTempIsectsKernel = api.createKernel("initIntersection", temp_isects);
        this.compactIsectsKernel = api.createKernel("compactIntersection", this.isects, temp_isects, prefixsum.getPredicate(), prefixsum.getPrefixSum());
        this.transferIsectsKernel = api.createKernel("transferIntersection", this.isects, temp_isects);

    }
    
    public void compactPixels(int globalSize)
    {
        api.execute1D(initTempPixelsKernel, globalSize, 1);
        api.execute1D(compactPixelsKernel, globalSize, 1);
        api.execute1D(transferPixelsKernel, globalSize, 1);
    }
    
    public void compactIsects(int globalSize)
    {
        api.execute1D(initTempIsectsKernel, globalSize, 1);
        api.execute1D(compactIsectsKernel, globalSize, 1);
        api.execute1D(transferIsectsKernel, globalSize, 1);
    }
    
    public void executePrefixSum()
    {
        prefixsum.execute();
    }
}
