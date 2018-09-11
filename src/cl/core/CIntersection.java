/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import coordinate.generic.raytrace.AbstractIntersection;
import org.jocl.struct.CLTypes.cl_float2;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CIntersection extends Struct implements AbstractIntersection{
    cl_float4 p;
    cl_float4 n;
    cl_float2 uv;
    int id;
    int hit;
    cl_float2 pixel;
}
