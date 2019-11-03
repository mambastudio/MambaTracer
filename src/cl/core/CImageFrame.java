/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import java.util.Arrays;
import wrapper.core.CBufferFactory;
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
    private CFloatBuffer rFrameBuffer;
    private CFloatBuffer rFrameCount;
    private CFloatBuffer rLogLuminance;    
    private CIntBuffer   rFrameARGB;
      
    //kernels
    private CKernel initFrameAccumKernel;
    private CKernel initFrameBufferKernel;
    private CKernel initFrameARGBKernel;
    private CKernel initLogLuminanceKernel;
    private CKernel averageAccumKernel;
    private CKernel updateFrameImageKernel;
    private CKernel logLuminanceKernel;
    
    private final int globalSize;
    private final int localSize = 1;
    
    private final CFloatScan scan;
    
    public CImageFrame(OpenCLPlatform configuration, int w, int h)
    {
        this.w = w; this.h = h;
        this.configuration = configuration;
        this.globalSize = w * h;   
        
        createBuffers();
        
        this.scan = new CFloatScan(configuration);
        this.scan.init(rLogLuminance);
        
        createKernels();
    }
    
    public final void createBuffers()
    {
        this.rFrameAccum    = CBufferFactory.allocFloat("frameAccum", configuration.context(), globalSize * 4, READ_WRITE);
        this.rFrameBuffer   = CBufferFactory.allocFloat("frameBuffer", configuration.context(), globalSize * 4, READ_WRITE);
        this.rFrameCount    = CBufferFactory.initFloatValue("frameCount", configuration.context(), configuration.queue(),1, READ_WRITE);
        this.rFrameARGB     = CBufferFactory.allocInt("frameARGB", configuration.context(), globalSize, READ_WRITE);
        this.rLogLuminance  = CBufferFactory.allocFloat("logLuminance", configuration.context(), globalSize, READ_WRITE);
        
        
        //this.rWidth         = CBufferFactory.initIntValue("rWidth", configuration.context(), configuration.queue(), w, READ_WRITE);
        //this.rHeight        = CBufferFactory.initIntValue("rHeight", configuration.context(), configuration.queue(), h, READ_WRITE);
    }
    
    public final void createKernels()
    {
        this.initFrameAccumKernel   = configuration.createKernel("initFloat4DataXYZ", rFrameAccum);
        this.initFrameBufferKernel  = configuration.createKernel("initFloat4DataXYZ", rFrameBuffer);
        this.initFrameARGBKernel    = configuration.createKernel("initIntDataRGB", rFrameARGB);
        this.initLogLuminanceKernel = configuration.createKernel("InitFloatData", rLogLuminance);
        this.averageAccumKernel     = configuration.createKernel("averageAccum", rFrameAccum, rFrameCount, rFrameBuffer);
        this.updateFrameImageKernel = configuration.createKernel("updateFrameImage", rFrameBuffer, rFrameARGB, scan.getTotalBufferCL(), scan.getTotalElementsBufferCL());
        this.logLuminanceKernel     = configuration.createKernel("logLuminance", rFrameBuffer, rLogLuminance);      
    }
    
    public void initBuffers()
    {
        configuration.queue().put1DRangeKernel(initFrameAccumKernel, globalSize, localSize);
        configuration.queue().put1DRangeKernel(initFrameARGBKernel, globalSize, localSize);
        configuration.queue().put1DRangeKernel(initFrameBufferKernel, globalSize, localSize);
        configuration.queue().put1DRangeKernel(initLogLuminanceKernel, globalSize, localSize);
        
        rFrameCount.mapWriteValue(configuration.queue(), 1);
    }
    
    public void processImage()
    {
        averageAccumulation();         
        calculateLogLuminance();
        
        scan.executeTotalElements();
        scan.executeTotalSum();
              
        updateFrameImage();
        
    }
    
    public void averageAccumulation()
    {
        configuration.queue().put1DRangeKernel(averageAccumKernel, globalSize, localSize);
    }
     
    public void calculateLogLuminance()
    {
        configuration.queue().put1DRangeKernel(logLuminanceKernel, globalSize, localSize);
    }
    
    public void updateFrameImage()
    {
        configuration.queue().put1DRangeKernel(updateFrameImageKernel, globalSize, localSize);
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
