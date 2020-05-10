/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.data.CPoint3;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class CShaderParameter extends ByteStruct{
    float base; //opacity, multiplier
    CPoint3 base_color; //r, g, b
    CPoint3 opacity;
    float nu, nv; //u v vector
    
    
    int type; //emitter, diffuse, reflection, refraction
    
    public CShaderParameter()
    {
        this.base = 1;
        this.base_color = new CPoint3(0.8f, 0.8f, 0.8f);
        this.opacity = new CPoint3(1, 1, 1);
    }
}
