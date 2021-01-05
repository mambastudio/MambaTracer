/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.struct.CIntersection;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.IntValue;
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
public final class CCompact {
    OpenCLConfiguration platform;
    
    int GLOBALSIZE =0;
    int LOCALSIZE = 100; 
    
    CMemory<CIntersection> origIsectBuffer;
    CMemory<CIntersection> tempIsectBuffer;    
    CMemory<IntValue> origPixels;
    CMemory<IntValue> tempPixels;
    
    CMemory<IntValue> totalCount;
    
    CKernel initTempIsectsKernel;
    CKernel initTempPixelsKernel;
    CKernel compactAtomicKernel;
    CKernel initOrigIsectsKernel;
    CKernel initOrigPixelsKernel; 
    CKernel tempToOriginalIsectsKernel;  
    CKernel tempToOriginalPixelsKernel;
    
    
    public CCompact(OpenCLConfiguration platform)
    {
        this.platform = platform;        
    }
    public void init(CMemory<CIntersection> isectBuffer, CMemory<IntValue> pixels, CMemory<IntValue> totalCount)
    {        
        this.GLOBALSIZE = isectBuffer.getSize();
        
        //init buffers
        this.origIsectBuffer    = isectBuffer;
        this.tempIsectBuffer    = platform.createBufferB(CIntersection.class, GLOBALSIZE, READ_WRITE);
        this.origPixels         = pixels;
        this.tempPixels         = platform.createBufferI(IntValue.class, GLOBALSIZE, READ_WRITE);
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
        totalCount.setCL(new IntValue(0));
        platform.execute1DKernel(initTempIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(initTempPixelsKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(compactAtomicKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(initOrigIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(initOrigPixelsKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(tempToOriginalIsectsKernel, GLOBALSIZE, LOCALSIZE);
        platform.execute1DKernel(tempToOriginalPixelsKernel, GLOBALSIZE, LOCALSIZE);
    }  
}
