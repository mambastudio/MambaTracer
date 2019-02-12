/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.data.struct.CIntersection;
import wrapper.core.CBufferFactory;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CStructBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class SimpleTest {
    public static void main(String... args)
    {
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(CLFileReader.readFile(SimpleTest.class, "cl\\CommonBase.cl")); 
        
        CStructBuffer<CIntersection> isectBuffer  = CBufferFactory.allocStruct("intersctions", configuration.context(), CIntersection.class, 1, READ_ONLY);
    }
}
