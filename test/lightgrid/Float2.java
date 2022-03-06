/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lightgrid;

import coordinate.struct.structfloat.FloatStruct;

/**
 *
 * @author user
 */
public class Float2 extends FloatStruct{
    public float x = 0;
    public float y = 0;
    
    public Float2()
    {
        x = 0;
        y = 0;
    }
    
    public Float2(float x, float y)
    {
        this.x = x;
        this.y = y;
    }
    
    public void set(float x, float y) {
        this.x = x;
        this.y = y;
        this.refreshGlobalArray();
    }
    
    @Override
    public String toString()
    {
        return String.format("(%3.5f, %3.5f)", x, y);
    }
}
