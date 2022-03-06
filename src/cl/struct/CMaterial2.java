/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.abstracts.MaterialInterface;
import coordinate.parser.attribute.MaterialT;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CMaterial2 extends Structure implements MaterialInterface<CMaterial2>  {
     //surface
    public CSurfaceParameter2 param; 
       

    public CMaterial2()
    {
        param = new CSurfaceParameter2();
    }
    
    public void setDiffuse(float r, float g, float b)
    {
        param.diffuse_color.set(r, g, b);
        this.refreshGlobalArray();
    }
    
    public void setEmitter(float r, float g, float b)
    {
        param.emission_color.set(r, g, b);
        param.emission_param.set('x', 1);
        param.emission_param.set('y', 15);
        this.refreshGlobalArray();
    }
    
    @Override
    public void setMaterial(CMaterial2 mat) {
        param = mat.param.copy();       
        this.refreshGlobalArray();
    }

    @Override
    public CMaterial2 copy() {
        CMaterial2 mat = new CMaterial2();
        mat.param = param.copy();
        return mat;
    }
    
    public void setSurfaceParameter(CSurfaceParameter2 param)
    {
        this.param = param;
    }

    @Override
    public void setMaterialT(MaterialT mat) {
        param.diffuse_color.set(mat.diffuse.r, mat.diffuse.g, mat.diffuse.b, mat.diffuse.w);        
        this.refreshGlobalArray();
    }
}
