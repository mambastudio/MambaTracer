/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lightgrid;

import cl.kernel.CSource;
import cl.struct.CLightGrid;
import coordinate.println.PrintFloat;
import coordinate.sampling.sat.SATSubgrid;
import java.util.Arrays;
import org.jocl.CL;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import static wrapper.core.memory.CLocalMemory.LOCALFLOAT;
import wrapper.core.memory.values.FloatValue;
import wrapper.core.memory.values.IntValue;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class TestLightGrid {
    public static void main(String... args)
    {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Print.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Util.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "LightGrid.cl"));
        
        CL.setExceptionsEnabled(true);
        //setup configuration
        OpenCLConfiguration configuration = OpenCLConfiguration.getDefault(new String[]{stringBuilder.toString()});
        
        
        //SAT CPU
        SATSubgrid subgrid = new SATSubgrid(16, 32, 1600, 1600);
        float[] arrFunc = new float[2560000];
        Arrays.fill(arrFunc, 1);      
        subgrid.setArray(arrFunc);
        
        //SAT GPU
        CMemory<CLightGrid> lightGrid = configuration.createBufferB(CLightGrid.class, 1, READ_WRITE);  
        lightGrid.mapWriteMemory(gridCL->{          
            CLightGrid grid = new CLightGrid(); //should be new if not initialized before in GPU
            float[] arr = new float[512];
            Arrays.fill(arr, 1);     
            grid.setToAllFunc(false, arr);
            gridCL.set(0, grid);
        });
      
        
        //Init GPU SAT
        CKernel prefixSumRowSubgridKernel = configuration.createKernel("prefixSumRowSubgrid", lightGrid, LOCALFLOAT);
        CKernel prefixSumColSubgridKernel = configuration.createKernel("prefixSumColSubgrid", lightGrid, LOCALFLOAT);
        
        configuration.execute1DKernel(prefixSumRowSubgridKernel, 2560000, 16);
        configuration.execute1DKernel(prefixSumColSubgridKernel, 2560000, 32);
                
        //Init variables
        
        //CPU
        int     subgridIndex    = 20;
        float   rand0           = (float) Math.random();
        float   rand1           = (float) Math.random();        
                
        float[] uv              = new float[2];
        float[] pdf             = new float[1];
        
        //GPU    
        CMemory<IntValue> sIndex    = configuration.createValueI(IntValue.class, new IntValue(subgridIndex), READ_WRITE);
        CMemory<Float2>     rand    = configuration.createValueF(Float2.class, new Float2(rand0, rand1), READ_WRITE);
        CMemory<Float2>     cuv     = configuration.createValueF(Float2.class, new Float2(), READ_WRITE);
        CMemory<FloatValue> cpdf    = configuration.createValueF(FloatValue.class, new FloatValue(), READ_WRITE);
        
        CKernel sampleSubgridTest   = configuration.createKernel("sampleSubgridTest", lightGrid, sIndex, rand, cuv, cpdf);
        
        //execute CPU
        subgrid.sampleContinuous(0, rand0, rand1, uv, pdf);
        //execute GPU
        configuration.execute1DKernel(sampleSubgridTest, 1, 1);
        
        //print
        System.out.println(Arrays.toString(uv));
        System.out.println(cuv.getCL());
        
        
        System.out.println(Arrays.toString(pdf));
        System.out.println(cpdf.getCL().v);    
        
        lightGrid.mapReadMemory(gridCL->{
            CLightGrid grid = gridCL.getCL();
            PrintFloat print = new PrintFloat(grid.sat);
            print.setDimension(1600, 1600);
            print.printArray(16, 32);
        });
    }
}
