/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CPoint3;
import coordinate.struct.annotation.arraysize;
import coordinate.struct.structbyte.Structure;


/**
 *
 * @author user
 */
public class CEnvironmentGrid extends Structure{
    public boolean      isPresent;    
    public int          width;
    public int          height;
    public CPoint3      cameraPosition;
    
    
    //int values encourages better sampling
    @arraysize(5000)
    public int[]        intLightGrid;   //100 * 50
    @arraysize(2560000)
    public int[]        intTileGrid; //16 * 32 * lightGrid

    //for temporary values
    @arraysize(5000)
    public float[]      floatLightGrid;   //100 * 50
    @arraysize(2560000)
    public float[]      floatTileGrid; //16 * 32 * lightGrid
    
    public void setWidth(int width)
    {
        this.width = width;
        this.refreshGlobalArray();
    }
    
    public void setHeight(int height)
    {
        this.height = height;
        this.refreshGlobalArray();
    }
    
    public void setCameraPosition(CPoint3 cameraPosition)
    {
        this.cameraPosition = cameraPosition;
        this.refreshGlobalArray();
    }
    
    public void setIsPresent(boolean isPresent)
    {
        this.isPresent = isPresent;
        this.refreshGlobalArray();
    }
}
