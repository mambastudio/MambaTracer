/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class CImageFrame {
    int w, h;
    OpenCLPlatform configuration;
    
    private CFloatBuffer rFrameAccum;  
    private CFloatBuffer rFrameCount;
    private CIntBuffer   rFrameARGB;
    private CFloatBuffer rWidth;
    private CFloatBuffer rHeight;
    
    //kernels
    private CKernel initFrameAccumKernel;
    private CKernel initFrameARGBKernel;
    private CKernel updateImageKernel;
    
    private final int globalSize;
    private final int localSize = 100;
  
    public CImageFrame(OpenCLPlatform configuration, int w, int h)
    {
        this.w = w; this.h = h;
        this.configuration = configuration;
        this.globalSize = w * h;   
        
        createBuffers();        
        createKernels();
    }
    
    public final void createBuffers()
    {
        this.rFrameAccum        = configuration.allocFloat("frameAccum", globalSize * 4, READ_WRITE);        
        this.rFrameCount        = configuration.allocFloatValue("frameCount",  1, READ_WRITE);
        this.rFrameARGB         = configuration.allocInt("frameARGB", globalSize, READ_WRITE);
        this.rWidth             = configuration.allocFloatValue("rWidth", w, READ_WRITE);
        this.rHeight            = configuration.allocFloatValue("rHeight", h, READ_WRITE);
        
    }
    
    public final void createKernels()
    {
        this.initFrameAccumKernel       = configuration.createKernel("initFloat4DataXYZ", rFrameAccum);        
        this.initFrameARGBKernel        = configuration.createKernel("initIntDataRGB", rFrameARGB);
        this.updateImageKernel          = configuration.createKernel("UpdateImageJitter", rFrameAccum, rFrameCount, rFrameARGB, rWidth, rHeight);
    }
    
    public void initBuffers()
    {
        configuration.queue().put1DRangeKernel(initFrameAccumKernel, globalSize, localSize);
        configuration.queue().put1DRangeKernel(initFrameARGBKernel, globalSize, localSize);      
        rFrameCount.mapWriteValue(configuration.queue(), 1);
    }
    
    public void processImage()
    {
        configuration.executeKernel1D(updateImageKernel, globalSize, localSize);
    }
    
    public void incrementFrameCount()
    {
        rFrameCount.mapWriteValue(configuration.queue(), rFrameCount.mapReadValue(configuration.queue()) + 1);
    }
    
    public CFloatBuffer getFrameCount()
    {
        return rFrameCount;
    }
    
    public CFloatBuffer getFrameAccum()
    {
        return rFrameAccum;
    }
    
    public CIntBuffer getFrameARGB()
    {
        return rFrameARGB;
    }
}
