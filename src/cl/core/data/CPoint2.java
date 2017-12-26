/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data;

import coordinate.generic.AbstractCoordinate;

/**
 *
 * @author user
 */
public class CPoint2 extends AbstractCoordinate
{
    public float x, y;
    
    public CPoint2() {
        super();
    }

    public CPoint2(float x, float y) {
        this.x = x;
        this.y = y;        
    }

    @Override
    public int getSize() {
        return 2;
    }

    @Override
    public float[] getArray() {
        return new float[]{x, y};
    }

    @Override
    public float get(char axis) {
        switch (axis) {
            case 'x':
                return x;
            case 'y':
                return y;
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.  
        }
    }

    @Override
    public void set(char axis, float value) {
        switch (axis) {
            case 'x':
                x = value;
                break;
            case 'y':
                y = value;
                break;
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    @Override
    public void set(float... values) {
        x = values[0];
        y = values[1];
    }

    @Override
    public void setIndex(int index, float value) {
        switch (index)
        {
            case 0:
                x = value;
                break;
            case 1:
                y = value;
                break;                
        }
    }    
}
