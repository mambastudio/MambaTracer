/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.data;

import coordinate.generic.SCoord;
import coordinate.struct.structfloat.FloatStruct;


/**
 *
 * @author user
 * 
 * This is only used for mesh data only, OpenCL handles data differently.
 * 
 */
public class CPoint3 extends FloatStruct implements SCoord<CPoint3, CVector3>{
public float x, y, z, w;
    public CPoint3(){
        super();
    }

    public CPoint3(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public CPoint3(CPoint3 p) {
        x = p.x;
        y = p.y;
        z = p.z;
    }
                
    public static final CVector3 sub(CPoint3 p1, CPoint3 p2) 
    {
        CVector3 dest = new CVector3();
        dest.x = p1.x - p2.x;
        dest.y = p1.y - p2.y;
        dest.z = p1.z - p2.z;
        return dest;
    }

    public static final CPoint3 mid(CPoint3 p1, CPoint3 p2) 
    {
        CPoint3 dest = new CPoint3();
        dest.x = 0.5f * (p1.x + p2.x);
        dest.y = 0.5f * (p1.y + p2.y);
        dest.z = 0.5f * (p1.z + p2.z);
        return dest;
    }
    
    @Override
    public CPoint3 setValue(float x, float y, float z) {
        CPoint3 p = SCoord.super.setValue(x, y, z);
        this.refreshGlobalArray();
        return p;
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
    public void set(float... values) {
        x = values[0];
        y = values[1];
        z = values[2];
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

    @Override
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
    public CPoint3 copy() {
        return new CPoint3(x, y, z);
    }

    @Override
    public CPoint3 getSCoordInstance() {
        return new CPoint3();
    }

    @Override
    public CVector3 getVCoordInstance() {
        return new CVector3();
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
    public String toString()
    {
        float[] array = getArray();
        return String.format("(%3.2f, %3.2f, %3.2f)", array[0], array[1], array[2]);
    }

    @Override
    public int getByteSize() {
        return 4;
    }
}
