/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.material;

import cl.core.CMaterialInterface;
import coordinate.parser.attribute.Color4T;
import coordinate.parser.attribute.MaterialT;
import java.io.Serializable;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class JMaterial extends Struct {
     
    public cl_float4 diffuse;         //diffuse    - r, g, b, w (pad)
    public float diffuseWeight;     //diffuse    - diffuse weight
    
    public cl_float4 reflection;      //reflection - r, g, b, w (pad)
    public float eu, ev, ior;       //reflection - eu, ev, ior
    public boolean iorEnabled;
    
    public cl_float4 emitter;
    public boolean emitterEnabled;
}
