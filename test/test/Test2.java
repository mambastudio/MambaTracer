/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import coordinate.utility.StructInfo;
import cl.core.data.struct.CBSDF;
import cl.core.data.struct.CFrame;
import cl.core.data.struct.CPath;
import java.util.Arrays;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class Test2 {
    public static void main(String... args)
    {                 
        Struct.showLayout(Path.class);
        
        StructInfo info = new StructInfo(CPath.class);
        System.out.println(Arrays.toString(info.offsets()));
    }
    
    
    public static class Path extends Struct
    {
        public cl_float4 throughput;
        public boolean active;
        public BSDF bsdf;
    }
    
    public static class BSDF extends Struct
    {
        public int materialID;              //material id
        public Frame frame;                //local frame of reference
        public cl_float4 localDirFix;       //incoming (fixed) incoming direction, in local
    }
    
    public static class Frame extends Struct
    {
        public cl_float4 mX;
        public cl_float4 mY;
        public cl_float4 mZ;
    }
}
