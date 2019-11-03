/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data;

import coordinate.generic.AbstractCoordinateInteger;

/**
 *
 * @author user
 */
public class CInt4 implements AbstractCoordinateInteger{
    public int x, y, z, w;
    
    public CInt4()
    {
        x = y = z = w = 0;
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
        throw new UnsupportedOperationException("Not supported yet. Make necessary changes"); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setIndex(int index, int value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getSize() {
        return 4;
    }

    @Override
    public int[] getArray() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getByteSize() {
        return 4;
    }
    
    
}
