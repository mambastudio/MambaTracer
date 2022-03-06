/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CPoint3;
import coordinate.sampling.sat.SATSubgrid;
import coordinate.struct.annotation.arraysize;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CLightGrid extends Structure{
    public boolean isPresent; //is light present
    
    public int sampleTile;
    
    //to remove later
    public int       width;
    public int       height;
    
    //full region
    public int nu, nv;
    public int subgridRangeX, subgridRangeY; //local size (16, 32)
    
    public CPoint3 cameraPosition;
    
    //16 * 32 * lightGrid(100 * 50)
    @arraysize(2560000)
    public float accum[];
    
    //16 * 32 * lightGrid(100 * 50)
    @arraysize(2560000)
    public float func[];

    //16 * 32 * lightGrid(100 * 50)
    @arraysize(2560000)
    public float sat[];
    
    //no point of deriving the same functions
    private final SATSubgrid subgrid;
    
    public CLightGrid()
    {
        isPresent = false;
        sampleTile = 0;
        nu = 1600;
        nv = 1600;
        subgridRangeX = 16;
        subgridRangeY = 32;
        cameraPosition = new CPoint3();
        
        subgrid = new SATSubgrid(16, 32, 1600, 1600);
    }  
    public void setIsPresent(boolean value)
    {
        this.isPresent = value;
        this.refreshGlobalArray();
    }
    
    public void setSampleTile(boolean value)
    {
        this.sampleTile = value ? 1 : 0;
        this.refreshGlobalArray();
    }
    
    public void setCameraPosition(CPoint3 cPosition)
    {
        this.cameraPosition = cPosition;
        this.refreshGlobalArray();
    }
    
    public void setFunc(boolean isPresent, float... array)
    {
        this.isPresent = isPresent;
        if(array == null || array.length != func.length)
            throw new IllegalStateException("array state is wrong");
        System.arraycopy(array, 0, func, 0, array.length);
        this.refreshGlobalArray();
    }
    
    public void setToAllFunc(boolean isPresent, float... array)
    {
        this.isPresent = isPresent;
        if(array == null || array.length != subgrid.subgridArea())
            throw new IllegalStateException("array state is wrong");
        for(int subgridI = 0; subgridI<subgrid.subgridCount(); subgridI++)
        {
            for(int i = 0; i<array.length; i++)
            {
                int index = subgrid.globalIndexInSubgrid(subgridI, i);
                func[index] = array[i];
            }
        }
        this.refreshGlobalArray();
    }
    
    //to remove later
    public void setSize(int width, int height)
    {
        this.width = width;
        this.height = height;
        
        this.refreshGlobalArray();
    }
}
