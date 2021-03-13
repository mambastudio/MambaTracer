/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import static cl.abstracts.MambaAPIInterface.getGlobal;
import cl.data.CColor4;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.FloatValue;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CImage {
    private final OpenCLConfiguration configuration;
    private int width = 100, height = 100;
    
    private int globalSize;
    private int localSize;
    private int frameSize;
    
    private CMemory<FloatValue> cFrameCount;             //frame count
    private CMemory<CColor4>    cFrameAccum;             //accumulate luminance
    private CMemory<CColor4>    cFrameBuffer;            //average luminance = accum/count
    private CMemory<IntValue>   cFrameARGB;              //final argb image (after tonemapping)
    private CMemory<IntValue>   cFrameSize;    
    private CMemory<FloatValue> cTotalLogLuminance;
    private CMemory<IntValue>   cTotalNumber;
    
    private CMemory<FloatValue> cloglw;                   //log luminance of every pixel  
    private CMemory<IntValue>   cloglwcount;              //predicate (0, 1) of log luminance i.e. loglw > 0 ? 1:0
    
    private CMemory<FloatValue> cgamma;
    private CMemory<FloatValue> cexposure;
    
    //kernels    
    private CKernel initFrameAccumKernel;
    private CKernel initFrameARGBKernel;        
    private CKernel averageAccum1Kernel;    
    private CKernel updateImageFrameKernel;
    
    private PrefixSumFloat prefixSumLoglw;
    private PrefixSumInteger prefixSumloglwCount;
    
    public CImage(OpenCLConfiguration configuration)
    {
        this.configuration = configuration;
        
        createWorkItemSizes();
    }
    
    public CImage(OpenCLConfiguration configuration, int width, int height)
    {
        this(configuration);
        this.width = width;
        this.height = height;
        
        create();
    }
    
    public final void create()
    {
        createWorkItemSizes();
        createBuffers();
        createKernels();
    }
    
    public final void createWorkItemSizes()
    {        
        localSize = 128;
        globalSize = getGlobal(width * height, localSize);
        frameSize = width * height;
    }
    
    public final void createBuffers()
    {
        this.cFrameCount            = configuration.createFromF(FloatValue.class,  new float[]{1}, READ_WRITE);
        this.cFrameAccum            = configuration.createBufferF(CColor4.class, frameSize, READ_WRITE);       
        this.cFrameBuffer           = configuration.createBufferF(CColor4.class, frameSize, READ_WRITE);
        this.cFrameARGB             = configuration.createBufferI(IntValue.class, frameSize, READ_WRITE);
        
        this.cFrameSize             = configuration.createFromI(IntValue.class, new int[]{frameSize}, READ_WRITE);
    
        this.cloglw                 = configuration.createBufferF(FloatValue.class, frameSize, READ_WRITE);
        this.cloglwcount            = configuration.createBufferI(IntValue.class, frameSize, READ_WRITE);
        
        this.prefixSumLoglw         = new PrefixSumFloat(configuration, cloglw);
        this.prefixSumloglwCount    = new PrefixSumInteger(configuration, cloglwcount);
        
        this.cTotalLogLuminance     = prefixSumLoglw.getCTotal();
        this.cTotalNumber           = prefixSumloglwCount.getCTotal();
        
        this.cgamma                 = configuration.createValueF(FloatValue.class, new FloatValue(2.2f), READ_WRITE);
        this.cexposure              = configuration.createValueF(FloatValue.class, new FloatValue(0.18f), READ_WRITE);
        
    }
    
    public final void createKernels()
    {
        this.initFrameAccumKernel       = configuration.createKernel("InitFloat4DataXYZ", cFrameAccum);        
        this.initFrameARGBKernel        = configuration.createKernel("InitIntDataRGB", cFrameARGB);   //introduce this kernel     
        
        this.averageAccum1Kernel        = configuration.createKernel("averageAccum", cFrameAccum, cFrameCount, cFrameBuffer, cloglw, cloglwcount, cFrameSize );
        this.updateImageFrameKernel     = configuration.createKernel("updateFrameImage", cFrameBuffer, cFrameARGB, cTotalLogLuminance, cTotalNumber, cFrameSize, cexposure, cgamma);
    }
    
    
    public void initBuffers()
    {
        configuration.execute1DKernel(initFrameAccumKernel, globalSize, localSize);
        configuration.execute1DKernel(initFrameARGBKernel, globalSize, localSize);     
        cFrameCount.setCL(new FloatValue(1));
    }
    
    public void processImage()
    {
        configuration.execute1DKernel(averageAccum1Kernel, globalSize, localSize);
        
        //no need to calculate after 3 frames/loops
        if(cFrameCount.getCL().v < 3)
        {
            this.prefixSumLoglw.execute();
            this.prefixSumloglwCount.execute();
        }
        configuration.execute1DKernel(updateImageFrameKernel, globalSize, localSize);
        
    }
            
    public void updateImageFrameKernel()
    {
        configuration.execute1DKernel(updateImageFrameKernel, globalSize, localSize);
    }
    
    public void setGamma(float value)
    {
        cgamma.setCL(new FloatValue(value));
    }
    
    public float getGamma()
    {
        return cgamma.getCL().v;
    }
    
    public void setExposure(float value)
    {
        cexposure.setCL(new FloatValue(value));
    }
    
    public float getExposure()
    {
        return cexposure.getCL().v;
    }
    
    public void initFrameCount()
    {
        cFrameCount.setCL(new FloatValue(1));
    }
    
    public void incrementFrameCount()
    {
        cFrameCount.setCL(new FloatValue(cFrameCount.getCL().v + 1));
    }
    
    public CMemory<IntValue> getCSize()
    {
        return cFrameSize;
    }
    
    public CMemory<FloatValue> getFrameCount()
    {
        return cFrameCount;
    }
    
    public CMemory<CColor4> getFrameAccum()
    {
        return cFrameAccum;
    }
    
    public CMemory<IntValue> getFrameARGB()
    {
        return cFrameARGB;
    }
}
