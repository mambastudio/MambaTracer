/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.struct.CIntersection;
import java.util.Arrays;
import wrapper.core.CBufferFactory;
import wrapper.core.CBufferMemory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;

/**
 *
 * @author user
 */
public final class CCompaction {    
    private final OpenCLPlatform configuration;   
    private int size;
    private final int LOCALSIZECONSTANT = 256;
    
    private int gSize1, gSize2, gSize3;
    private int lSize1, lSize2, lSize3;
    
    private CIntBuffer sum_level1, sum_level1_length;       
    private CIntBuffer sum_level2, sum_level2_length;
    private CIntBuffer sum_level3, sum_level3_length;
    
    private CKernel scanKernel1, scanKernel2, scanKernel3;
    private CKernel sumgKernel2, sumgKernel1;
    
    private CKernel resetTempIntersection, compactIntersection, transferIntersection;
    
    private CStructBuffer<CIntersection> isectBuffer;
    private CStructBuffer<CIntersection> tempIsectBuffer;
    
    public CCompaction(OpenCLPlatform configuration)
    {
        this.configuration = configuration;       
    }
    
    public void init(CStructBuffer<CIntersection> isectBuffer)
    {
        this.size = isectBuffer.getSize();
        this.isectBuffer = isectBuffer;
        this.tempIsectBuffer = CBufferFactory.allocStruct("temp_intersctions", configuration.context(), CIntersection.class, isectBuffer.getSize(), READ_ONLY);
        initCBuffers(size);
        initCKernels();
    }
    
    private void initCBuffers(int size)
    {
        gSize1              = length1(size);
        lSize1              = localsize(gSize1);                    
        gSize2              = length2(size);
        lSize2              = localsize(gSize2);
        gSize3              = length3(size);
        lSize3              = localsize(gSize3);
        
        sum_level1            = configuration.allocInt("sum_level1", size, READ_WRITE);
        sum_level1_length     = configuration.allocIntValue("sum_level1_length", size, READ_WRITE);
        sum_level2            = configuration.allocInt("sum_level2", gSize2, READ_WRITE);
        sum_level2_length     = configuration.allocIntValue("sum_level2_length", gSize2, READ_WRITE);
        sum_level3            = configuration.allocInt("sum_level3", gSize3, READ_WRITE);
        sum_level3_length     = configuration.allocIntValue("sum_level3_length", gSize3, READ_WRITE);        
    }
    
    private void initCKernels()
    {
        scanKernel1    = configuration.createKernel("blelloch_scan_isect_g" , isectBuffer, sum_level1, sum_level2, sum_level1_length, CBufferMemory.LOCALINT);  
        scanKernel2    = configuration.createKernel("blelloch_scan_g"       , sum_level2,  sum_level3, sum_level2_length, CBufferMemory.LOCALINT);
        scanKernel3    = configuration.createKernel("blelloch_scan"         , sum_level3,  sum_level3_length, CBufferMemory.LOCALINT);        
        sumgKernel2    = configuration.createKernel("add_groups"            , sum_level2,  sum_level3);
        sumgKernel1    = configuration.createKernel("add_groups_n"          , sum_level1,  sum_level2, sum_level1_length);        
        
        resetTempIntersection   = configuration.createKernel("resetIntersection"          , tempIsectBuffer);       
        compactIntersection     = configuration.createKernel("compactIntersection"        , isectBuffer, tempIsectBuffer, sum_level1);       
        transferIntersection    = configuration.createKernel("transferIntersection"       , isectBuffer, tempIsectBuffer);      
    }
    
    public void execute()
    {
        configuration.executeKernel1D(scanKernel1, gSize1, lSize1);  
        configuration.executeKernel1D(scanKernel2, gSize2, lSize2);       
        configuration.executeKernel1D(scanKernel3, gSize3, lSize3);       
        configuration.executeKernel1D(sumgKernel2, gSize2, lSize2);
        configuration.executeKernel1D(sumgKernel1, gSize1, lSize1);
        
        configuration.executeKernel1D(resetTempIntersection,    size, 1);
        configuration.executeKernel1D(compactIntersection,      size, 1);
        configuration.executeKernel1D(transferIntersection,     size, 1);
    }
    
    public int log2( int bits ) // returns 0 for bits=0
    {
        int log = 0;
        if( ( bits & 0xffff0000 ) != 0 ) { bits >>>= 16; log = 16; }
        if( bits >= 256 ) { bits >>>= 8; log += 8; }
        if( bits >= 16  ) { bits >>>= 4; log += 4; }
        if( bits >= 4   ) { bits >>>= 2; log += 2; }
        return log + ( bits >>> 1 );
    }
        
    public int pow2length(int length)
    {
        int log2 = log2(length);
        int difference = (int)(Math.pow(2, log2)) - length;
        
        if(difference == 0) return length;
        else                return (int) Math.pow(2, log2+1);
    }
    
    public int length1(int size)
    {
        int length = pow2length(size);
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 3);
        if(full_length == 0)
            return 1;
        else if(length > full_length)
            return full_length;
        else
            return length;
    }
    
    public int length2(int size)
    {
        int length = length1(size); length /= LOCALSIZECONSTANT;
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 2);
        if(length > full_length)
            return full_length;
        else
            return length;
    }
    
    public int length3(int size)
    {
        int length = length2(size); length /= LOCALSIZECONSTANT;
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 1);
        if(length > full_length)
            return full_length;
        else
            return length;
    }    
    
    public int localsize(int size)
    {
        int local = pow2length(size);
        return local > LOCALSIZECONSTANT ? LOCALSIZECONSTANT : local;
    }    
}
