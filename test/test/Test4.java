/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import coordinate.utility.StructInfo;
import java.util.Arrays;
import org.jocl.struct.CLTypes.cl_float2;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class Test4 {
    public static void main(String... args)
    {
        Struct.showLayout(Intersect.class);
        
    }
    
    public static class Intersect extends Struct
    {
        public cl_float2 b;
        public int c;     
        public cl_float4 d;
        public int e;
        public int f;
    }
}
