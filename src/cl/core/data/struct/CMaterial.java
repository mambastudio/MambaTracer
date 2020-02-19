/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.CMaterialInterface;
import cl.core.data.CColor4;
import coordinate.parser.attribute.Color4T;
import coordinate.parser.attribute.MaterialT;
import coordinate.struct.ByteStruct;
import java.io.Serializable;

/**
 *
 * @author user
 */
public class CMaterial extends ByteStruct implements Serializable, CMaterialInterface{
    public CColor4 diffuse;         //diffuse    - r, g, b, w (pad)
    public float diffuseWeight;     //diffuse    - diffuse weight
    
    public CColor4 reflection;      //reflection - r, g, b, w (pad)
    public float eu, ev, ior;       //reflection - eu, ev, ior
    public boolean iorEnabled;      //reflection - transmission enabled
    
    public CColor4 emitter;         //emission   - r, g, b, w (power)
    public boolean emitterEnabled;  //emission   - emission enabled
    
    public CMaterial()
    {
        this.diffuse = new CColor4(1, 1, 1);
        this.diffuseWeight = 0;
        
        this.reflection = new CColor4();
        this.eu = this.ev = this.ior = 0;
        this.iorEnabled = false;
        
        this.emitter = new CColor4();
        this.emitterEnabled = false;
    }
    
    public void setDiffuse(float r, float g, float b)
    {
        diffuse.set(r, g, b, 1);
        this.refreshGlobalArray();
    }
    
    public void setEmitter(float r, float g, float b)
    {
        emitter.set(r, g, b, 1);
        this.refreshGlobalArray();
    }
    
    public void setEmitter(Color4T color)
    {
        setEmitter(color.r, color.g, color.b);
    }
    
    public void setDiffuse(float r, float g, float b, float w)
    {
        diffuse.set(r, g, b, w);
        this.refreshGlobalArray();
    }    
       
    private Color4T getColor4T(CColor4 v)
    {
        return new Color4T(v.get('r'), v.get('g'), v.get('b'), v.get('w'));
    }
    
    @Override
    public void setMaterial(MaterialT mat) {
        diffuse.set(mat.diffuse.r, mat.diffuse.g, mat.diffuse.b, mat.diffuse.w);
        diffuseWeight = mat.diffuseWeight;
       
        reflection.set(mat.reflection.r, mat.reflection.g, mat.reflection.b, mat.reflection.w);
        eu = mat.eu; ev = mat.ev; ior = mat.ior; iorEnabled = mat.iorEnabled;
        
        emitter.set(mat.emitter.r, mat.emitter.g, mat.emitter.b, mat.emitter.w);
        emitterEnabled = mat.emitterEnabled;
        this.refreshGlobalArray();
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
    
    public void setEmitterEnabled(boolean b) {
        this.emitterEnabled = b;
        this.refreshGlobalArray();
    }
}
