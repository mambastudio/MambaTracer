/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.data;

import coordinate.generic.AbstractCoordinateInteger;
import coordinate.struct.structint.IntStruct;

/**
 *
 * @author user
 */
public class CInt3 extends IntStruct implements AbstractCoordinateInteger
{
    public int x, y, z;
    
    public CInt3()
    {
        x = y = z = 0;
    }

    @Override
    public int get(char axis) {
        switch (axis) {
            case 'x':
                return x;
            case 'y':
                return y;
            case 'z':
                return z;           
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.  
        }
    }

    @Override
    public void set(char axis, int value) {
        switch (axis) {
            case 'x':
                x = value;
                break;
            case 'y':
                y = value;
                break;
            case 'z':
                z = value;
                break;            
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    @Override
    public void set(int... values) {
        x = values[0];
        y = values[1];
        z = values[2];        
    }

    @Override
    public void setIndex(int index, int value) {
        switch (index)
        {
            case 0:
                x = value;
                break;
            case 1:
                y = value;
                break;    
            case 2:
                z = value;
                break;            
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    @Override
    public int getSize() {
        return 4;
    }

    @Override
    public int[] getArray() {
        return new int[]{x, y, z, 0};
    }

    @Override
    public int getByteSize() {
        return 4;
    }
}
