/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

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
public class CMaterial extends Struct implements Serializable, CMaterialInterface{   
   
    public cl_float4 diffuse;         //diffuse    - r, g, b, w (pad)
    public float diffuseWeight;     //diffuse    - diffuse weight
    
    public cl_float4 reflection;      //reflection - r, g, b, w (pad)
    public float eu, ev, ior;       //reflection - eu, ev, ior
    public boolean iorEnabled;      //reflection - transmission enabled
    
    public cl_float4 emitter;         //emission   - r, g, b, w (power)
    public boolean emitterEnabled;  //emission   - emission enabled
    
    
    public CMaterial()
    {
        set(diffuse, 0.95f, 0.95f, 0.95f);
    }
    
    public void setDiffuse(float r, float g, float b)
    {
        set(diffuse, r, g, b);
    }
    
    public void setDiffuse(float r, float g, float b, float w)
    {
        set(diffuse, r, g, b, w);
    }    
    
    private void set(cl_float4 v, float r, float g, float b)
    {
        v.set(0, r);
        v.set(1, g);
        v.set(2, b);
    }
    
    private void set(cl_float4 v, float r, float g, float b, float w)
    {
        v.set(0, r);
        v.set(1, g);
        v.set(2, b);
        v.set(3, w);
    }
    
    private Color4T getColor4T(cl_float4 v)
    {
        return new Color4T(v.get(0), v.get(1), v.get(2), v.get(3));
    }
    
    private void set(cl_float4 v, Color4T color)
    {
        v.set(0, color.r);
        v.set(1, color.g);
        v.set(2, color.b);
    }

    @Override
    public void setMaterial(MaterialT mat) {
        set(diffuse, mat.diffuse);
        diffuseWeight = mat.diffuseWeight;
        
        set(reflection, mat.reflection);
        eu = mat.eu; ev = mat.ev; ior = mat.ior; iorEnabled = mat.iorEnabled;
        
        set(emitter, mat.emitter);
        emitterEnabled = mat.emitterEnabled;
    }

    @Override
    public MaterialT getMaterial() {
        MaterialT mat = new MaterialT();
        mat.diffuse = getColor4T(diffuse);
        mat.diffuseWeight = diffuseWeight;
        
        mat.reflection = getColor4T(reflection);
        mat.eu = eu; mat.ev = ev; mat.ior = ior; mat.iorEnabled = iorEnabled;
        
        mat.emitter = getColor4T(emitter);
        mat.emitterEnabled = emitterEnabled;
        return mat;
    }
}
