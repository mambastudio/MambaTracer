/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.ploc;

import cl.core.data.struct.CNode;
import cl.core.data.struct.array.CStructIntArray;
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
public class ButterflySort {
    OpenCLPlatform configuration = null;
    
    CStructIntArray<CNode> cdata   = null;
    CIntBuffer clength = null;
    CFloatBuffer cpowerx = null;
    
    public ButterflySort(OpenCLPlatform configuration, CStructIntArray<CNode> data, int length)
    {
        this.configuration = configuration;
        this.cdata = data; 
        clength = configuration.allocIntValue("lengthSize", length, READ_ONLY);
        cpowerx = configuration.allocFloatValue("powerX", 0, READ_WRITE);
    }
    
    public void sort()
    {
        //start of sort structure
        int radix  = 2;        
        int until = until(clength.get(0)); 
        int T = (int) (Math.pow(radix, until)/radix);//data.length/radix if n is power of 2;
        
        System.out.println(T);
        
        int globalSize = T;
        int localSize = globalSize<256 ? globalSize : 256;
        
        //kernel initialization
        CKernel cbutterfly1    = configuration.createKernel("butterfly1", cdata.getCBuffer(), clength, cpowerx);
        CKernel cbutterfly2    = configuration.createKernel("butterfly2", cdata.getCBuffer(), clength, cpowerx);
    
        for(int xout = 1; xout<=until; xout++)
        {     
            configuration.setFloatValue((float)Math.pow(radix, xout), cpowerx); //PowerX = (Math.pow(radix, xout));      
                                    
            // OpenCL kernel call
            configuration.executeKernel1D(cbutterfly1, globalSize, localSize); 
            
            if(xout > 1)
            {                
                for(int xin = xout; xin > 0; xin--)
                {
                    configuration.setFloatValue((float)Math.pow(radix, xin), cpowerx); //PowerX = (Math.pow(radix, xin));
                    
                    // OpenCL kernel call
                    configuration.executeKernel1D(cbutterfly2, globalSize, localSize); 
                    
                }
            }
        }
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
    
    public int until(int length)
    {
        int log2 = log2(length);
        int difference = (int)(Math.pow(2, log2)) - length;
        
        if(difference == 0) return log2;
        else                return log2+1;
    }
}
