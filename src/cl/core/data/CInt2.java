/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data;


import coordinate.generic.AbstractCoordinateInteger;
import java.util.Arrays;

/**
 *
 * @author user
 */
public class CInt2 implements AbstractCoordinateInteger{
    
    public int x, y;
    
    public CInt2()
    {
        
    }

    @Override
    public int get(char axis) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void set(char axis, int value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void set(int... values) {
        x = values[0];
        y = values[1];
    }

    @Override
    public void setIndex(int index, int value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getSize() {
        return 2;
    }

    @Override
    public int[] getArray() {
        return new int[]{x, y};
    }

    @Override
    public int getByteSize() {
        return 4;
    }
    
    
    @Override
    public String toString()
    {
        int[] array = getArray();
        return Arrays.toString(array);
    }
}
