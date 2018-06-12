/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.morton;

import cl.core.src.CLSource;
import org.jocl.CL;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class GPUMortonTest {
    public static void main(String... args)
    {
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2 = CLFileReader.readFile(GPUMortonTest.class, "GPUMorton.cl");
        
        CL.setExceptionsEnabled(true);
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(source1, source2);
        
        int mortonKarras[] = new int[]{150994944, 150994944, 153391689, 153391689, 301989888,301989888,306783378,306783378,603979776,603979776,613566756,613566756};        
        CIntBuffer mortons = CBufferFactory.wrapInt("mortons", configuration.context(), configuration.queue(), mortonKarras, READ_ONLY);
        CIntBuffer numofprims = CBufferFactory.initIntValue("numofprims", configuration.context(), configuration.queue(), mortonKarras.length, READ_ONLY);
        
        CKernel testMortonKernel = configuration.program().createKernel("testMortons", mortons, numofprims); 
        configuration.queue().put1DRangeKernel(testMortonKernel, 1, 1);
    }
}
