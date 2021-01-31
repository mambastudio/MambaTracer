/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.struct.CSurfaceParameter;
import cl.abstracts.MaterialInterface;
import static cl.abstracts.MaterialInterface.BRDFType.DIFFUSE;
import static cl.abstracts.MaterialInterface.BRDFType.EMITTER;
import coordinate.parser.attribute.MaterialT;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CMaterial extends Structure implements MaterialInterface<CMaterial> {
    public CSurfaceParameter param1; //surface 1
    public CSurfaceParameter param2; //surface 2
       
    //portal?
    public boolean isPortal;
    
    //either fresnel, texture, or value = (0, 1, 2);
    public int opacityType;
    
    public float fresnel;
    public float opacity;
    
    public CMaterial()
    {
        param1 = new CSurfaceParameter();
        param2 = new CSurfaceParameter();
        
        isPortal = false;
        
        opacityType = 2;
        fresnel = 1.2f;
        opacity = 1f;
        
        
    }
    
    public void setEmitterEnabled(boolean emit)
    {
        if(emit)
            param1.brdfType = EMITTER.ordinal();
        else
            param1.brdfType = DIFFUSE.ordinal();
        this.refreshGlobalArray();
    }
    
    public boolean isEmitterEnabled()
    {
        return param1.brdfType == EMITTER.ordinal();
    }
    
    public void setDiffuse(float r, float g, float b)
    {
        param1.base_color.set(r, g, b, 1);
        this.refreshGlobalArray();
    }
    
    public void setEmitter(float r, float g, float b)
    {
        param1.emission_color.set(r, g, b, 1);
        this.refreshGlobalArray();
    }
    
    public void setEmitterPower(float power)
    {
        param1.emission = power;
        this.refreshGlobalArray();
    }
    
    public void setOpacity(float opacity)
    {
        this.opacity = opacity;
        this.refreshGlobalArray();
    }
    
    public void setMaterial(MaterialT mat) {
        param1.base_color.set(mat.diffuse.r, mat.diffuse.g, mat.diffuse.b, mat.diffuse.w);
        param1.diffuse_roughness = mat.diffuseWeight;
        param1.brdfType = DIFFUSE.ordinal();
       
        param1.emission_color.set(mat.emitter.r, mat.emitter.g, mat.emitter.b, mat.emitter.w);        
        this.refreshGlobalArray();
    }

    @Override
    public void setMaterial(CMaterial m) {
        param1 = m.param1.copy();
        param2 = m.param2.copy();
        isPortal = m.isPortal;
        opacityType = m.opacityType;
        fresnel = m.fresnel;
        opacity = m.opacity;
        
        this.refreshGlobalArray();
    }
    
    public void setSurfaceParameter1(CSurfaceParameter param)
    {
        param1 = param;
    }
    
    public void setSurfaceParameter2(CSurfaceParameter param)
    {
        param2 = param;
    }
    
    public void setSurfaceParameter(CSurfaceParameter param1, CSurfaceParameter param2)
    {
        setSurfaceParameter1(param1);
        setSurfaceParameter2(param2);
    }

    @Override
    public CMaterial copy() {
        CMaterial mat = new CMaterial();
        mat.param1 = param1.copy();
        mat.param2 = param2.copy();
        mat.isPortal = isPortal;
        mat.opacityType = opacityType;
        mat.fresnel = fresnel;
        mat.opacity = opacity;
        return this;
    }

    @Override
    public void setMaterialT(MaterialT t) {
        
        param1.base_color.set(t.diffuse.r, t.diffuse.g, t.diffuse.b);
        param1.diffuse_roughness = t.diffuseWeight;
        param1.emission_color.set(t.emitter.r, t.emitter.g, t.emitter.b);
        //either emission or diffuse
        param1.brdfType = t.emitterEnabled ?  EMITTER.ordinal() : DIFFUSE.ordinal();
        this.refreshGlobalArray();
    }
}
