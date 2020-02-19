/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 * 
 * I might have considered prefix sum here but honestly, the issues with barriers
 * which is not hardware agnostic wasn't a good consideration and performance degradation
 * was experienced due to alot of kernel calls.
 * 
 * Some algorithms avoid barriers but still require a lot of kernel calls in which doesn't
 * give any merit if atomics are used.
 * 
 * Hence atomics are handled here instead
 * 
 * This is where wavefront for path tracing is considered.
 * 
 */
public final class SCompact {
    OpenCLPlatform platform;
    
    int GLOBALSIZE =0;
    int LOCALSIZE = 100; 
    
    CStructTypeBuffer<SIsect> origIsectBuffer;
    CStructTypeBuffer<SIsect> tempIsectBuffer;    
    CIntBuffer origPixels;
    CIntBuffer tempPixels;
    
    CIntBuffer totalCount;
    
    CKernel initTempIsectsKernel;
    CKernel initTempPixelsKernel;
    CKernel compactAtomicKernel;
    CKernel initOrigIsectsKernel;
    CKernel initOrigPixelsKernel; 
    CKernel tempToOriginalIsectsKernel;  
    CKernel tempToOriginalPixelsKernel;
    
    
    public SCompact(OpenCLPlatform platform)
    {
        this.platform = platform;        
    }
    public void init(CStructTypeBuffer<SIsect> isectBuffer, CIntBuffer pixels, CIntBuffer totalCount)
    {        
        this.GLOBALSIZE = isectBuffer.getSize();
        
        //init buffers
        this.origIsectBuffer    = isectBuffer;
        this.tempIsectBuffer    = platform.allocStructType("tempIsect", SIsect.class, GLOBALSIZE, READ_WRITE);
        this.origPixels         = pixels;
        this.tempPixels         = platform.allocInt("tempPixels", GLOBALSIZE, READ_WRITE);
        this.totalCount         = totalCount;
               
        //init kernels
        initTempIsectsKernel        = platform.createKernel("InitIntersection", tempIsectBuffer);
        initTempPixelsKernel        = platform.createKernel("InitIntData", tempPixels);
        tempToOriginalIsectsKernel  = platform.createKernel("TransferIntersection",origIsectBuffer, tempIsectBuffer);
        tempToOriginalPixelsKernel  = platform.createKernel("TransferPixels", origPixels, tempPixels);                
        initOrigIsectsKernel        = platform.createKernel("InitIntersection", origIsectBuffer);
        initOrigPixelsKernel        = platform.createKernel("InitIntData", origPixels);
        compactAtomicKernel         = platform.createKernel("CompactAtomic", origIsectBuffer, tempIsectBuffer, origPixels, tempPixels, totalCount);
    }
    
    public void execute()
    {
        totalCount.mapWriteValue(platform.queue(), 0);
        platform.executeKernel1D(initTempIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(initTempPixelsKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(compactAtomicKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(initOrigIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(initOrigPixelsKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(tempToOriginalIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.executeKernel1D(tempToOriginalPixelsKernel, GLOBALSIZE, LOCALSIZE);
    }  
}
