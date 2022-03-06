/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CColor4;
import cl.data.CPoint3;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CSurfaceParameter2 extends Structure {
    //this surface is done by texture
    public boolean          isDiffuseTexture;
    public boolean          isGlossyTexture;
    public boolean          isRoughnessTexture;
    public boolean          isMirrorTexture;
    
    //brdf parameters
    public CColor4          diffuse_color;
    public CPoint3          diffuse_param;
    public CColor4          glossy_color;
    public CPoint3          glossy_param;
    public CColor4          mirror_color;
    public CPoint3          mirror_param;
    public CColor4          emission_color;
    public CPoint3          emission_param;
    
    public CSurfaceParameter2()
    {
        isDiffuseTexture = false;
        isGlossyTexture = false;
        isRoughnessTexture = false;
        isMirrorTexture = false;
        
        diffuse_color   = new CColor4(0.95f, 0.95f, 0.95f);
        diffuse_param   = new CPoint3(1, 0, 0);
        glossy_color    = new CColor4(0.95f, 0.95f, 0.95f);
        glossy_param    = new CPoint3();
        mirror_color    = new CColor4(0.95f, 0.95f, 0.95f);
        mirror_param    = new CPoint3();
        emission_color  = new CColor4(1f, 1f, 1f);
        emission_param  = new CPoint3();
    }
    
    public CSurfaceParameter2 copy()
    {
        CSurfaceParameter2 param    = new CSurfaceParameter2();
        param.isDiffuseTexture      = isDiffuseTexture;
        param.isGlossyTexture       = isGlossyTexture;
        param.isRoughnessTexture    = isRoughnessTexture;
        param.isMirrorTexture       = isMirrorTexture;
        
        param.diffuse_color         = diffuse_color.copy();
        param.diffuse_param         = diffuse_param.copy(); 
        param.glossy_color          = glossy_color .copy(); 
        param.glossy_param          = glossy_param.copy();  
        param.mirror_color          = mirror_color.copy();  
        param.mirror_param          = mirror_param.copy();  
        param.emission_color        = emission_color.copy();
        param.emission_param        = emission_param.copy();
        
        return param;
    }
}
