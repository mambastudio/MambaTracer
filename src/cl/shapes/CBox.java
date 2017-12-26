/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.shapes;

import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CBox extends Struct{
    public cl_float4 min;
    public cl_float4 max;    
    
    
    public CBox()
    {
        max.set(0,  1); max.set(1,  1); max.set(2,  1);
        min.set(0, -1); min.set(1, -1); min.set(2, -1);
    }
}
