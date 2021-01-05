/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.core.data.CColor4;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class CSurfaceParameter extends ByteStruct {
    //this surface is done by texture
    public boolean          isTexture;
    public int              numTexture;
    
    //what brdf (DIFFUSE = 0, EMITTER = 1)
    public int              brdfType;
            
    //brdf parameters
    public CColor4          base_color;
    public float            diffuse_roughness;
    public CColor4          specular_color;
    public float            specular_IOR;
    public float            metalness;
    public float            transmission;
    public CColor4          transmission_color;
    public float            emission;
    public CColor4          emission_color;
    public float            anisotropy_nx;
    public float            anisotropy_ny;
    
    public CSurfaceParameter()
    {
        isTexture = false;
        numTexture = 0;


        brdfType = 0;


        base_color = new CColor4(0.95f, 0.95f, 0.95f);
        diffuse_roughness = 0;
        specular_color = new CColor4(0.95f, 0.95f, 0.95f);
        specular_IOR = 0;
        metalness = 0;
        transmission = 0;
        transmission_color = new CColor4(0.95f, 0.95f, 0.95f);
        emission = 20;
        emission_color = new CColor4(1, 1, 1);
        anisotropy_nx = 0;
        anisotropy_ny = 0;
    }
    
    public CSurfaceParameter copy()
    {
        CSurfaceParameter param = new CSurfaceParameter();
        param.isTexture             = isTexture;
        param.numTexture            = numTexture; 
        param.brdfType              = brdfType;
        param.base_color            = base_color.copy();
        param.diffuse_roughness     = diffuse_roughness;
        param.specular_color        = specular_color.copy();
        param.specular_IOR          = specular_IOR;
        param.metalness             = metalness;
        param.transmission          = transmission;
        param.transmission_color    = transmission_color.copy();
        param.emission              = emission;
        param.emission_color        = emission_color.copy();
        param.anisotropy_nx         = anisotropy_nx;
        param.anisotropy_ny         = anisotropy_ny ;
        return param;
    }
}
