/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lightgrid;

import cl.kernel.CSource;
import coordinate.println.PrintFloat;
import coordinate.struct.annotation.arraysize;
import coordinate.struct.structbyte.Structure;
import java.util.Arrays;
import org.jocl.CL;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import static wrapper.core.memory.LocalMemory.LOCALFLOAT;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class TestSimpleGrid {
    public static void main(String... args)
    {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Print.cl"));
        stringBuilder.append(CLFileReader.readFile(TestSimpleGrid.class, "SimpleGrid.cl"));
        
        CL.setExceptionsEnabled(true);
        //setup configuration
        OpenCLConfiguration configuration = OpenCLConfiguration.getDefault(new String[]{stringBuilder.toString()});
    
        CMemory<CSimpleGrid> lightGrid = configuration.createBufferB(CSimpleGrid.class, 1, READ_WRITE);  
        lightGrid.mapWriteMemory(gridCL->{          
            CSimpleGrid grid = new CSimpleGrid(); //should be new if not initialized before in GPU
            float[] arr = new float[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01f, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };            
            grid.setFunc(arr);
            gridCL.set(0, grid);
        });
      
        
        //Init GPU SAT
        CKernel prefixSumRowSubgridKernel = configuration.createKernel("prefixSumRowSubgrid", lightGrid, LOCALFLOAT);
        CKernel prefixSumColSubgridKernel = configuration.createKernel("prefixSumColSubgrid", lightGrid, LOCALFLOAT);
        
        configuration.execute1DKernel(prefixSumRowSubgridKernel, 256, 16);
        configuration.execute1DKernel(prefixSumColSubgridKernel, 256, 16);
        
        lightGrid.mapReadMemory(gridCL->{
            CSimpleGrid grid = gridCL.getCL();
            PrintFloat print = new PrintFloat(grid.sat);
            print.setPrecision(8, 4);
            print.setDimension(16, 16);
            print.printArray();
        });
    }
    
    public static class CSimpleGrid extends Structure
    {
        public int nu, nv;

        //16 * 16
        @arraysize(256)
        public float func[];

        //16 * 16
        @arraysize(256)
        public float sat[];
        
        public CSimpleGrid()                
        {
            this.nu = 16;
            this.nv = 16;
        }
        
        public void setFunc(float... array)
        {        
            if(array == null || array.length != func.length)
                throw new IllegalStateException("array state is wrong");
            System.arraycopy(array, 0, func, 0, array.length);
            this.refreshGlobalArray();
        }
    }
    
    
}


