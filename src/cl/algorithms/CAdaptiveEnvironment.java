/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.data.CPoint3;
import cl.struct.CLightGrid;
import coordinate.sampling.sat.SAT;
import java.util.Arrays;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import static wrapper.core.memory.CLocalMemory.LOCALFLOAT;

/**
 *
 * @author user
 */
public class CAdaptiveEnvironment 
{
    private final OpenCLConfiguration configuration;
    
    private final CMemory<CLightGrid> lightGrid;
        
    private final CKernel prefixSumRowSubgridKernel;
    private final CKernel prefixSumColSubgridKernel;    
    private final CKernel initLightGridKernel;
    private final CKernel calculateFuncKernel;
    private final CKernel initFuncAndSatKernel;
    
    private SAT sat;
    private final int satCountSample = 500;
    
    public CAdaptiveEnvironment(OpenCLConfiguration configuration)
    {
        this.configuration = configuration;
        this.lightGrid     = configuration.createBufferB(CLightGrid.class, 1, READ_WRITE); 
        lightGrid.mapWriteMemory(gridCL->{          
            CLightGrid grid = new CLightGrid(); //should be new if not initialized before in GPU (in this case, here)
            float[] arr = new float[512];
            Arrays.fill(arr, 1);     
            grid.setToAllFunc(false, arr);             
            gridCL.set(0, grid);            
        });
       
        this.prefixSumRowSubgridKernel  = configuration.createKernel("prefixSumRowSubgrid", lightGrid, LOCALFLOAT);
        this.prefixSumColSubgridKernel  = configuration.createKernel("prefixSumColSubgrid", lightGrid, LOCALFLOAT);    
        this.initLightGridKernel        = configuration.createKernel("initLightGrid", lightGrid);
        this.calculateFuncKernel        = configuration.createKernel("calculateFunc", lightGrid);
        this.initFuncAndSatKernel       = configuration.createKernel("initFuncAndSat", lightGrid);
    }
    
    public CMemory<CLightGrid> getLightGridCL()
    {
        return lightGrid;
    }
    
    public void setCameraPosition(CPoint3 position)
    {
        lightGrid.mapWriteMemory(grid->{
            CLightGrid cgrid = grid.getCL();
            cgrid.setCameraPosition(position);
        });
    }
    
    public void setHDRLuminance(SAT sat)
    {       
        this.sat = sat;
        
        //clear buffers
        clearBuffers(); //clearBuffers() tends to set isPresent = false;
      
        //set HDR luminance to light grid
        float[] luminanceArr = sat.sampleToGrid(satCountSample, 16, 32);
        lightGrid.mapWriteMemory(gridCL->{
            CLightGrid grid = gridCL.getCL(); //should be new if not initialized before in GPU    
            grid.setSize(sat.getNu(), sat.getNv());
            grid.setToAllFunc(true, luminanceArr);
            gridCL.set(0, grid);
        });
        
        //calculate sat
        configuration.execute1DKernel(prefixSumRowSubgridKernel, 2560000, 16);
        configuration.execute1DKernel(prefixSumColSubgridKernel, 2560000, 32);
    }
    
    public void removeHDRLuminance()
    {
        sat = null;
        clearBuffers();
    }
    
    public void resetHDRLuminance()
    {
        //clear buffers
        clearBuffers();
        
        setIsPresent(true); //clearBuffers() tends to set isPresent = false;
        
        if(sat == null)
        {
            setIsPresent(false);
            return;
        }
        
        //set HDR luminance to light grid
        float[] luminanceArr = this.sat.sampleToGrid(satCountSample, 16, 32);
        lightGrid.mapWriteMemory(gridCL->{
            CLightGrid grid = gridCL.getCL(); //should be new if not initialized before in GPU 
            grid.setSize(sat.getNu(), sat.getNv());
            grid.setToAllFunc(true, luminanceArr);
            gridCL.set(0, grid);
        });
        
        //calculate sat
        configuration.execute1DKernel(prefixSumRowSubgridKernel, 2560000, 16);
        configuration.execute1DKernel(prefixSumColSubgridKernel, 2560000, 32);
    }
    
    public void clearBuffers()
    {
        configuration.execute1DKernel(initLightGridKernel, 2560000, 128);        
        setIsPresent(false);
    }
    
    public void clearEverything()
    {
        configuration.execute1DKernel(initLightGridKernel, 2560000, 128);       
        setIsPresent(false);
    }
    
    public void update()
    {
        //clear func and sat array
        configuration.execute1DKernel(initFuncAndSatKernel, 2560000, 128);
        //calculate new func 
        configuration.execute1DKernel(calculateFuncKernel, 2560000, 128);        
        //calculate new sat
        configuration.execute1DKernel(prefixSumRowSubgridKernel, 2560000, 16);
        configuration.execute1DKernel(prefixSumColSubgridKernel, 2560000, 32);
        
    }
    
    public void setSampleTile(boolean value)
    {
        this.lightGrid.mapWriteMemory(gridCL->{
            CLightGrid grid = gridCL.getCL(); //should be new if not initialized before in GPU 
            grid.setSampleTile(value);
        });
    }
    
    public void setIsPresent(boolean value)
    {
        this.lightGrid.mapWriteMemory(gridCL->{
            CLightGrid grid = gridCL.getCL(); //should be new if not initialized before in GPU 
            grid.setIsPresent(value);
        });
    }
    
    public boolean isPresent()
    {
        CLightGrid gridCL = this.lightGrid.getCL();
        return gridCL.isPresent;
    }
    
}
