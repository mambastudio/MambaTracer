/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.struct.CIntersection;
import wrapper.core.CBufferFactory;
import wrapper.core.CBufferMemory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public final class CCompaction {    
    private final OpenCLPlatform configuration;   
    private int size;
    private final int LOCALSIZECONSTANT = 16;
    
    private CIntBuffer total = null;
    private CIntBuffer array_size = null;
    
    private int gSize1, gSize2, gSize3, gSize4, gSize5, gSize6, gSize7, gSize8, gSize9, gSize10;
    private int lSize1, lSize2, lSize3, lSize4, lSize5, lSize6, lSize7, lSize8, lSize9, lSize10;
    
    private CIntBuffer sum_level1, sum_level1_length;       //sum_leveln_length should be deleted, not useful
    private CIntBuffer sum_level2, sum_level2_length;
    private CIntBuffer sum_level3, sum_level3_length;
    private CIntBuffer sum_level4, sum_level4_length;       
    private CIntBuffer sum_level5, sum_level5_length;
    private CIntBuffer sum_level6, sum_level6_length;
    private CIntBuffer sum_level7, sum_level7_length;       
    private CIntBuffer sum_level8, sum_level8_length;
    private CIntBuffer sum_level9, sum_level9_length;
    private CIntBuffer sum_level10, sum_level10_length;
    
    private CKernel scanKernel1, scanKernel2, scanKernel3;
    private CKernel scanKernel4, scanKernel5, scanKernel6;
    private CKernel scanKernel7, scanKernel8, scanKernel9;
    private CKernel scanKernel10;
    
    private CKernel sumgKernel9, sumgKernel8, sumgKernel7, sumgKernel6, sumgKernel5, sumgKernel4, sumgKernel3, sumgKernel2, sumgKernel1;
    
    //optional kernel
    private CKernel totalIntersectionKernel;
    private CKernel processIsectDataKernel;
    private CKernel initIntArrayKernel;
    private CKernel resetTempIntersection, compactIntersection, transferIntersection;
    
    private CStructTypeBuffer<CIntersection> isectBuffer;
    private CStructTypeBuffer<CIntersection> tempIsectBuffer;
    
    public CCompaction(OpenCLPlatform configuration)
    {
        this.configuration = configuration;       
    }
    
    public void init(CStructTypeBuffer<CIntersection> isectBuffer, CIntBuffer total)
    {
        this.size = isectBuffer.getSize();
        this.total = total;
        this.isectBuffer = isectBuffer;        
        initCBuffers(size);
        initCKernels();
    }
    
    private void initCBuffers(int size)
    {
        gSize1              = length(size, 1);
        lSize1              = localsize(gSize1);                    
        gSize2              = length(size, 2);
        lSize2              = localsize(gSize2);
        gSize3              = length(size, 3);
        lSize3              = localsize(gSize3);
        gSize4              = length(size, 4);
        lSize4              = localsize(gSize4);                    
        gSize5              = length(size, 5);
        lSize5              = localsize(gSize5);
        gSize6              = length(size, 6);
        lSize6              = localsize(gSize6);
        gSize7              = length(size, 7);
        lSize7              = localsize(gSize7);                    
        gSize8              = length(size, 8);
        lSize8              = localsize(gSize8);
        gSize9              = length(size, 9);
        lSize9              = localsize(gSize9);
        gSize10             = length(size, 10);
        lSize10             = localsize(gSize10);
        
        this.tempIsectBuffer = CBufferFactory.allocStructType("temp_intersections", configuration.context(), CIntersection.class, this.size, READ_ONLY);
        this.array_size           = configuration.allocIntValue("array_size", size, READ_ONLY);
        
                
        sum_level1            = configuration.allocInt("sum_level1", gSize1, READ_WRITE);
        sum_level1_length     = configuration.allocIntValue("sum_level1_length", gSize1, READ_WRITE);
        sum_level2            = configuration.allocInt("sum_level2", gSize2, READ_WRITE);
        sum_level2_length     = configuration.allocIntValue("sum_level2_length", gSize2, READ_WRITE);
        sum_level3            = configuration.allocInt("sum_level3", gSize3, READ_WRITE);
        sum_level3_length     = configuration.allocIntValue("sum_level3_length", gSize3, READ_WRITE);   
        sum_level4            = configuration.allocInt("sum_level4", gSize4, READ_WRITE);
        sum_level4_length     = configuration.allocIntValue("sum_level4_length", gSize4, READ_WRITE);
        sum_level5            = configuration.allocInt("sum_level5", gSize5, READ_WRITE);
        sum_level5_length     = configuration.allocIntValue("sum_level5_length", gSize5, READ_WRITE);
        sum_level6            = configuration.allocInt("sum_level6", gSize6, READ_WRITE);
        sum_level6_length     = configuration.allocIntValue("sum_level6_length", gSize6, READ_WRITE);
        sum_level7            = configuration.allocInt("sum_level7", gSize7, READ_WRITE);
        sum_level7_length     = configuration.allocIntValue("sum_level7_length", gSize7, READ_WRITE);
        sum_level8            = configuration.allocInt("sum_level8", gSize8, READ_WRITE);
        sum_level8_length     = configuration.allocIntValue("sum_level8_length", gSize8, READ_WRITE);
        sum_level9            = configuration.allocInt("sum_level9", gSize9, READ_WRITE);
        sum_level9_length     = configuration.allocIntValue("sum_level9_length", gSize9, READ_WRITE);
        sum_level10           = configuration.allocInt("sum_level10", gSize10, READ_WRITE);
        sum_level10_length    = configuration.allocIntValue("sum_level10_length", gSize10, READ_WRITE);
    }
    
    private void initCKernels()
    {
        initIntArrayKernel      = configuration.createKernel("initIntArray", sum_level1);
        processIsectDataKernel  = configuration.createKernel("processIsectData", isectBuffer, sum_level1);
                        
        scanKernel1    = configuration.createKernel("blelloch_scan_g"       , sum_level1,   sum_level2,  sum_level1_length, CBufferMemory.LOCALINT);  
        scanKernel2    = configuration.createKernel("blelloch_scan_g"       , sum_level2,   sum_level3,  sum_level2_length, CBufferMemory.LOCALINT);
        scanKernel3    = configuration.createKernel("blelloch_scan_g"       , sum_level3,   sum_level4,  sum_level3_length, CBufferMemory.LOCALINT);  
        scanKernel4    = configuration.createKernel("blelloch_scan_g"       , sum_level4,   sum_level5,  sum_level4_length, CBufferMemory.LOCALINT);
        scanKernel5    = configuration.createKernel("blelloch_scan_g"       , sum_level5,   sum_level6,  sum_level5_length, CBufferMemory.LOCALINT);  
        scanKernel6    = configuration.createKernel("blelloch_scan_g"       , sum_level6,   sum_level7,  sum_level6_length, CBufferMemory.LOCALINT);
        scanKernel7    = configuration.createKernel("blelloch_scan_g"       , sum_level7,   sum_level8,  sum_level7_length, CBufferMemory.LOCALINT);  
        scanKernel8    = configuration.createKernel("blelloch_scan_g"       , sum_level8,   sum_level9,  sum_level8_length, CBufferMemory.LOCALINT);
        scanKernel9    = configuration.createKernel("blelloch_scan_g"       , sum_level9,   sum_level10, sum_level9_length, CBufferMemory.LOCALINT);          
        scanKernel10   = configuration.createKernel("blelloch_scan"         , sum_level10,  sum_level10_length, CBufferMemory.LOCALINT); 
        sumgKernel9    = configuration.createKernel("add_groups"            , sum_level9,  sum_level10);
        sumgKernel8    = configuration.createKernel("add_groups"            , sum_level8,  sum_level9);
        sumgKernel7    = configuration.createKernel("add_groups"            , sum_level7,  sum_level8);
        sumgKernel6    = configuration.createKernel("add_groups"            , sum_level6,  sum_level7);
        sumgKernel5    = configuration.createKernel("add_groups"            , sum_level5,  sum_level6);
        sumgKernel4    = configuration.createKernel("add_groups"            , sum_level4,  sum_level5);
        sumgKernel3    = configuration.createKernel("add_groups"            , sum_level3,  sum_level4);
        sumgKernel2    = configuration.createKernel("add_groups"            , sum_level2,  sum_level3);
        sumgKernel1    = configuration.createKernel("add_groups_n"          , sum_level1,  sum_level2, sum_level1_length);
        
        resetTempIntersection   = configuration.createKernel("resetIntersection"          , tempIsectBuffer);       
        compactIntersection     = configuration.createKernel("compactIntersection"        , isectBuffer, tempIsectBuffer, sum_level1);       
        transferIntersection    = configuration.createKernel("transferIntersection"       , isectBuffer, tempIsectBuffer);      
        
        totalIntersectionKernel = configuration.createKernel("totalIntersection", isectBuffer, sum_level1, array_size, total);
    }
        
    public void execute()
    {
        configuration.executeKernel1D(initIntArrayKernel,       gSize1, lSize1);
        configuration.executeKernel1D(processIsectDataKernel,   size,   1);  
        
        configuration.executeKernel1D(scanKernel1,  gSize1,  lSize1);        
        configuration.executeKernel1D(scanKernel2,  gSize2,  lSize2);       
        configuration.executeKernel1D(scanKernel3,  gSize3,  lSize3);   
        configuration.executeKernel1D(scanKernel4,  gSize4,  lSize4);        
        configuration.executeKernel1D(scanKernel5,  gSize5,  lSize5);       
        configuration.executeKernel1D(scanKernel6,  gSize6,  lSize6);   
        configuration.executeKernel1D(scanKernel7,  gSize7,  lSize7);        
        configuration.executeKernel1D(scanKernel8,  gSize8,  lSize8);       
        configuration.executeKernel1D(scanKernel9,  gSize9,  lSize9); 
        configuration.executeKernel1D(scanKernel10, gSize10, lSize10);
        configuration.executeKernel1D(sumgKernel9,  gSize9,  lSize9);
        configuration.executeKernel1D(sumgKernel8,  gSize8,  lSize8);
        configuration.executeKernel1D(sumgKernel7,  gSize7,  lSize7);
        configuration.executeKernel1D(sumgKernel6,  gSize6,  lSize6);
        configuration.executeKernel1D(sumgKernel5,  gSize5,  lSize5);
        configuration.executeKernel1D(sumgKernel4,  gSize4,  lSize4);
        configuration.executeKernel1D(sumgKernel3,  gSize3,  lSize3);
        configuration.executeKernel1D(sumgKernel2,  gSize2,  lSize2);
        configuration.executeKernel1D(sumgKernel1,  gSize1,  lSize1);
        
        configuration.executeKernel1D(totalIntersectionKernel, 1, 1);
        
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
    
    
    public int length(int size, int level)
    {
        int length = pow2length(size);
        
        length /= (int)Math.pow(LOCALSIZECONSTANT, level - 1);
        
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, log2(length));
        if(full_length == 0)
            return 1;
        else if(length > full_length)
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
