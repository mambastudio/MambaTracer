/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.compact;

import cl.core.CCompaction;
import cl.core.data.struct.CIntersection;
import cl.core.kernel.CLSource;
import java.util.Random;
import org.jocl.CL;
import wrapper.core.CBufferFactory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class TestCompact {
    public static void main(String... args)
    {
        CL.setExceptionsEnabled(true);
        
        int isectSize                           = 18;
        
        String source1                          = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2                          = CLFileReader.readFile(CLSource.class, "ScanCompact.cl");
        OpenCLPlatform configuration            = OpenCLPlatform.getDefault(source1, source2);
        
        CStructTypeBuffer<CIntersection> isects = CBufferFactory.allocStructType("intersections", configuration.context(), CIntersection.class, isectSize, READ_WRITE);
        CIntBuffer hitCount                     = CBufferFactory.initIntValue("hitCount", configuration.context(), configuration.queue(), isectSize, READ_WRITE);
        CCompaction compactIsect                = new CCompaction(configuration);
        
        Random r = new Random();
        isects.mapWriteBuffer(configuration.queue(), intersections -> {
          
            for (CIntersection intersection : intersections) {
                intersection.setHit(r.ints(isectSize, 0, 2).limit(1).findFirst().getAsInt());
                System.out.println(intersection.hit);
            }
        });
        
        System.out.println();
        System.out.println("Sorted");
        
        compactIsect.init(isects, hitCount);
        compactIsect.execute();
        
        isects.mapReadBuffer(configuration.queue(), intersections-> {
            for(CIntersection isect : intersections)
                System.out.println(isect.hit);
        });        
    }    
}
