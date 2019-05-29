/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.sampling;

import cl.core.kernel.CLSource;
import java.math.BigInteger;
import java.util.Random;
import wrapper.core.CBufferFactory;
import wrapper.core.CCommandQueue;
import wrapper.core.CConfiguration;
import wrapper.core.CContext;
import wrapper.core.CDevice;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.CPlatform;
import wrapper.core.CProgram;
import wrapper.core.CResourceFactory;
import wrapper.core.buffer.CIntBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class Test {
    public static void main(String... args)
    {
        rngTest();
    }
    
    public static void rngTest()
    {
        CPlatform platform = CConfiguration.getDefault();
        CDevice device = platform.getDeviceCPU();        
        CContext context = platform.createContext(device);
                    
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2 = CLFileReader.readFile(CLSource.class, "Sampling.cl");
        String source3 = CLFileReader.readFile(Test.class, "TestSampling.cl");
        
        CProgram program = context.createProgram(source1, source2, source3);
        CCommandQueue queue = context.createCommandQueue(device);        
        
        CIntBuffer cseed = CBufferFactory.allocInt("seed", context, 2, READ_ONLY);
        cseed.mapWriteBuffer(queue, buffer -> {
            buffer.put(BigInteger.probablePrime(30, new Random()).intValue());
            buffer.put(BigInteger.probablePrime(30, new Random()).intValue());
        });
        int globalSize = 20;
        
         //execute kernel
        CKernel kernel = program.createKernel("testRNG");
        kernel.putArgs(cseed);        
        queue.put1DRangeKernel(kernel, globalSize, 1);   
        
        CResourceFactory.releaseAll();
    }
}
