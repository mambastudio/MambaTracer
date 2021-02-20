/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algorithm;

import cl.algorithms.PrefixSumFloat;
import cl.kernel.CSource;
import java.util.Random;
import org.jocl.CL;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.FloatValue;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class PrefixSumFloatTest {
    public static void main(String... args)
    {
        CL.setExceptionsEnabled(true);
        //setup configuration
        OpenCLConfiguration configuration = OpenCLConfiguration.getDefault(CLFileReader.readFile(CSource.class, "Util.cl"));
    
        int length = 300; //any length is allowed
        double[] doubleData = new Random().doubles(length, 0, 2).toArray(); //random [1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
        float[] data = doublesToFloat(doubleData);
        
        CMemory<FloatValue> cpredicate        = configuration.createFromF(FloatValue.class, data, READ_WRITE);
        
        
        PrefixSumFloat prefixSum = new PrefixSumFloat(configuration, cpredicate);
        prefixSum.execute();
        prefixSum.printlnResults();
       
    }
    
    private static float[] doublesToFloat(double[] array) {
        float[] inFloatForm = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            inFloatForm[i] = (float) array[i];
        }//from ww w .jav  a 2 s .  c o m
        return inFloatForm;
    }
}
