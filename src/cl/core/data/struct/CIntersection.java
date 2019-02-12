/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import coordinate.generic.raytrace.AbstractIntersection;
import org.jocl.struct.CLTypes.cl_float2;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CIntersection extends Struct implements AbstractIntersection{
    public cl_float4 p;
    public cl_float4 n;
    public cl_float4 d;
    public cl_float2 uv;
    public int mat;
    public int id;
    public int hit;   
    public cl_float2 pixel;
}
