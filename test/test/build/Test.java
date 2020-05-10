/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.build;

import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import coordinate.parser.obj.OBJParser;
import java.math.BigInteger;
import java.util.Random;
import org.jocl.CL;
import wrapper.core.OpenCLPlatform;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class Test {
    
    //"C:\\Users\\user\\Documents\\3D Scenes\\mori_knob\\testObj.obj"
    //"C:\\Users\\user\\Documents\\3D Scenes\\Ajax\\Ajax_wall_emitter.obj"
    //"C:\\Users\\user\\Documents\\3D Scenes\\charger_free\\charger_emitter.obj"
    public static void main(String... args)
    {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(CLFileReader.readFile(CLSource.class, "Print.cl"));       
        stringBuilder.append(CLFileReader.readFile(CLSource.class, "Common.cl"));
        stringBuilder.append(CLFileReader.readFile(CLSource.class, "Primitive.cl"));
        stringBuilder.append(CLFileReader.readFile(CLSource.class, "NormalBVH.cl"));
        stringBuilder.append(CLFileReader.readFile(CLSource.class, "NormalBVHBuilder.cl"));
        
        CL.setExceptionsEnabled(true);
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(stringBuilder.toString());
        
        CMesh mesh = new CMesh(configuration);           
        OBJParser parser = new OBJParser(); 
        parser.read("C:\\Users\\user\\Documents\\3D Scenes\\plane\\Triangles.obj", mesh);
        mesh.initCLBuffers();
        
        GPUBuildBVH build = new GPUBuildBVH(configuration);
        build.build(mesh);
    }
    
}
