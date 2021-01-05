/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.data;

import coordinate.generic.VCoord;
import coordinate.struct.FloatStruct;

/**
 *
 * @author user
 */
public class CVector3 extends FloatStruct implements VCoord<CVector3>{
    public float x, y, z;
    
    public CVector3(){}
    public CVector3(float x, float y, float z){this.x = x; this.y = y; this.z = z;};
    public CVector3(CVector3 v) {this.x = v.x; this.y = v.y; this.z = v.z;}
        
    @Override
    public CVector3 getCoordInstance() {
        return new CVector3();
    }

    @Override
    public CVector3 copy() {
        return new CVector3(x, y, z);
    }

    @Override
    public float get(char axis) {
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

    public void set(char axis, float value) {
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
    public void set(float... values) {        
        x = values[0];
        y = values[1];
        z = values[2];
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
            case 2:
                z = value;
                break;
        }
    }

    @Override
    public int getSize() {
        return 4;
    }

    @Override
    public float[] getArray() {
        return new float[]{x, y, z, 0};
    }
      
    @Override
    public String toString()
    {
        float[] array = getArray();
        return String.format("(%8.2f, %8.2f, %8.2f)", array[0], array[1], array[2]);
    }

    @Override
    public int getByteSize() {
        return 4;
    }
}
