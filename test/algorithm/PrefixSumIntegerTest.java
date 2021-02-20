/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algorithm;

import cl.algorithms.PrefixSumInteger;
import cl.kernel.CSource;
import java.util.Random;
import org.jocl.CL;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.IntValue;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class PrefixSumIntegerTest {
    public static void main(String... args)
    {
         CL.setExceptionsEnabled(true);
        //setup configuration
        OpenCLConfiguration configuration = OpenCLConfiguration.getDefault(CLFileReader.readFile(CSource.class, "Util.cl"));
    
        int length = 200; //any length is allowed
        int[] data = new Random().ints(length, 1, 2).toArray(); //random [1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
        
        CMemory<IntValue> cpredicate        = configuration.createFromI(IntValue.class, data, READ_WRITE);
        
        
        PrefixSumInteger prefixSum = new PrefixSumInteger(configuration, cpredicate);
        prefixSum.execute();
        prefixSum.printlnResults();
    }
}
