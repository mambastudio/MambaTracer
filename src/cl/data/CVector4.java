/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.data;

import coordinate.generic.VCoord;
import coordinate.struct.structfloat.FloatStruct;

/**
 *
 * @author user
 */
public class CVector4 extends FloatStruct implements VCoord<CVector4>{
    
    public float x, y, z, w;
    public CVector4(){}
    public CVector4(float x, float y, float z, float w){this.x = x; this.y = y; this.z = z; this.w = w;};
    public CVector4(CVector4 v) {this.x = v.x; this.y = v.y; this.z = v.z; this.w = v.w;}

    @Override
    public CVector4 getCoordInstance() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CVector4 copy() {
        return new CVector4(x, y, z, w);
    }

    @Override
    public void set(float... values) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getByteSize() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
