/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import wrapper.core.CBufferMemory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class CFloatScan {
    private final OpenCLPlatform configuration;   
    private int size;
    
    private final int LOCALSIZECONSTANT = 32;
    /*
       largest float buffer size it can process is LOCALSIZECONSTANT^10;
    
       Scan algorithm is the blelloch scan (exclusive)
    */
    
    
    private CFloatBuffer total = null;
    private CFloatBuffer totalElements = null;
    private CIntBuffer array_size = null;
    
    private int gSize1, gSize2, gSize3, gSize4, gSize5, gSize6, gSize7, gSize8, gSize9, gSize10;
    private int lSize1, lSize2, lSize3, lSize4, lSize5, lSize6, lSize7, lSize8, lSize9, lSize10;
    
    private CFloatBuffer sum_level1;  private CIntBuffer sum_level1_length;       //sum_leveln_length should be deleted, not useful
    private CFloatBuffer sum_level2;  
    private CFloatBuffer sum_level3;  
    private CFloatBuffer sum_level4;  
    private CFloatBuffer sum_level5;  
    private CFloatBuffer sum_level6;  
    private CFloatBuffer sum_level7; 
    private CFloatBuffer sum_level8; 
    private CFloatBuffer sum_level9;
    private CFloatBuffer sum_level10;
    
    private CKernel scanKernel1, scanKernel2, scanKernel3;
    private CKernel scanKernel4, scanKernel5, scanKernel6;
    private CKernel scanKernel7, scanKernel8, scanKernel9;
    private CKernel scanKernel10;
    
    private CKernel totalKernel;
    private CKernel totalElementsKernel;
    
    private CKernel initFloatArrayKernel;
    
    private CKernel copyFloatBufferKernel;
    private CKernel copyBackFloatBufferKernel;
    private CKernel copyDigitFloatBufferKernel;
    
    private CKernel sumgKernel9, sumgKernel8, sumgKernel7, sumgKernel6, sumgKernel5, sumgKernel4, sumgKernel3, sumgKernel2, sumgKernel1;
    
    private CFloatBuffer floatBuffer;
        
    public CFloatScan(OpenCLPlatform configuration)
    {
        this.configuration = configuration;       
    }
    
    public void init(CFloatBuffer floatBuffer)
    {
        this.size = floatBuffer.getBuffer().capacity();      
        this.floatBuffer = floatBuffer;        
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
        
        this.array_size       = configuration.allocIntValue("array_sizef", size, READ_ONLY);
        this.total            = configuration.allocFloatValue("totalf", 0, READ_WRITE);
        this.totalElements    = configuration.allocFloatValue("totalElementsf", 0, READ_WRITE);
               
        sum_level1            = configuration.allocFloat("sum_level1f", gSize1, READ_WRITE);
        sum_level1_length     = configuration.allocIntValue("sum_level1_lengthf", gSize1, READ_WRITE);
        sum_level2            = configuration.allocFloat("sum_level2f", gSize2, READ_WRITE);        
        sum_level3            = configuration.allocFloat("sum_level3f", gSize3, READ_WRITE);
        sum_level4            = configuration.allocFloat("sum_level4f", gSize4, READ_WRITE);        
        sum_level5            = configuration.allocFloat("sum_level5f", gSize5, READ_WRITE);
        sum_level6            = configuration.allocFloat("sum_level6f", gSize6, READ_WRITE);
        sum_level7            = configuration.allocFloat("sum_level7f", gSize7, READ_WRITE);
        sum_level8            = configuration.allocFloat("sum_level8f", gSize8, READ_WRITE);
        sum_level9            = configuration.allocFloat("sum_level9f", gSize9, READ_WRITE);
        sum_level10           = configuration.allocFloat("sum_level10f", gSize10, READ_WRITE);
    }
    
    private void initCKernels()
    {     
        initFloatArrayKernel = configuration.createKernel("InitFloatData", sum_level1);
        copyFloatBufferKernel = configuration.createKernel("copyFloatBuffer", sum_level1, floatBuffer);
        copyBackFloatBufferKernel = configuration.createKernel("copyFloatBuffer", floatBuffer, sum_level1);
        copyDigitFloatBufferKernel = configuration.createKernel("copyDigitFloatBuffer", sum_level1, floatBuffer);
        
        scanKernel1    = configuration.createKernel("blelloch_scan_g_f"       , sum_level1,  sum_level2, CBufferMemory.LOCALFLOAT);  
        scanKernel2    = configuration.createKernel("blelloch_scan_g_f"       , sum_level2,  sum_level3, CBufferMemory.LOCALFLOAT);
        scanKernel3    = configuration.createKernel("blelloch_scan_g_f"       , sum_level3,  sum_level4, CBufferMemory.LOCALFLOAT);  
        scanKernel4    = configuration.createKernel("blelloch_scan_g_f"       , sum_level4,  sum_level5, CBufferMemory.LOCALFLOAT);
        scanKernel5    = configuration.createKernel("blelloch_scan_g_f"       , sum_level5,  sum_level6, CBufferMemory.LOCALFLOAT);  
        scanKernel6    = configuration.createKernel("blelloch_scan_g_f"       , sum_level6,  sum_level7, CBufferMemory.LOCALFLOAT);
        scanKernel7    = configuration.createKernel("blelloch_scan_g_f"       , sum_level7,  sum_level8, CBufferMemory.LOCALFLOAT);  
        scanKernel8    = configuration.createKernel("blelloch_scan_g_f"       , sum_level8,  sum_level9, CBufferMemory.LOCALFLOAT);
        scanKernel9    = configuration.createKernel("blelloch_scan_g_f"       , sum_level9,  sum_level10,CBufferMemory.LOCALFLOAT);          
        scanKernel10   = configuration.createKernel("blelloch_scan_f"         , sum_level10, CBufferMemory.LOCALFLOAT); 
        sumgKernel9    = configuration.createKernel("add_groups_f"            , sum_level9,  sum_level10);
        sumgKernel8    = configuration.createKernel("add_groups_f"            , sum_level8,  sum_level9);
        sumgKernel7    = configuration.createKernel("add_groups_f"            , sum_level7,  sum_level8);
        sumgKernel6    = configuration.createKernel("add_groups_f"            , sum_level6,  sum_level7);
        sumgKernel5    = configuration.createKernel("add_groups_f"            , sum_level5,  sum_level6);
        sumgKernel4    = configuration.createKernel("add_groups_f"            , sum_level4,  sum_level5);
        sumgKernel3    = configuration.createKernel("add_groups_f"            , sum_level3,  sum_level4);
        sumgKernel2    = configuration.createKernel("add_groups_f"            , sum_level2,  sum_level3);
        sumgKernel1    = configuration.createKernel("add_groups_n_f"          , sum_level1,  sum_level2, sum_level1_length);
        
        totalKernel = configuration.createKernel("total_f", floatBuffer, sum_level1, array_size, total);
        totalElementsKernel = configuration.createKernel("totalElements_f", floatBuffer, sum_level1, array_size, totalElements);
    }
    
    public void executePrefixSum()
    {
        configuration.executeKernel1D(initFloatArrayKernel, gSize1, 1);
        configuration.executeKernel1D(copyFloatBufferKernel, size, 1);    
        
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
        configuration.executeKernel1D(totalKernel, 1, 1);
        
        configuration.executeKernel1D(copyBackFloatBufferKernel, size, 1);
    }
    
    public void executeTotalSum()
    {
        //initFloatBuffers.initAllBuffers();
        
        configuration.executeKernel1D(initFloatArrayKernel, gSize1, 1);
        configuration.executeKernel1D(copyFloatBufferKernel, size, 1);    
        
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
        configuration.executeKernel1D(totalKernel, 1, 1);
    }
    
    public void executeTotalElements()
    {
        //initFloatBuffers.initAllBuffers();
        
        configuration.executeKernel1D(initFloatArrayKernel, gSize1, 1);
        configuration.executeKernel1D(copyDigitFloatBufferKernel, size, 1);    
        
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
        configuration.executeKernel1D(totalElementsKernel, 1, 1);        
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
    
    public CFloatBuffer getTotalBufferCL()
    {
        return total;
    }
    
    public CFloatBuffer getTotalElementsBufferCL()
    {
        return totalElements;
    }
    
    public float getTotal()
    {
        return total.mapReadValue(configuration.queue());
    }
    
    public float getTotalElements()
    {
        return totalElements.mapReadValue(configuration.queue());
    }
}

