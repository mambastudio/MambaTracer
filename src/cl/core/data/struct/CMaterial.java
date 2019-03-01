/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CMaterial extends Struct{
    public cl_float4 diffuse;
    
    public CMaterial()
    {
        diffuse.set(0, 0.95f);
        diffuse.set(1, 0.95f);
        diffuse.set(2, 0.95f);
        diffuse.set(3, 1f);
    }
    
    public void setDiffuse(float r, float g, float b)
    {
        diffuse.set(0, r);
        diffuse.set(1, g);
        diffuse.set(2, b);
        diffuse.set(3, 1f);
    }
}
