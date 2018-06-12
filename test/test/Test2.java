/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.CNormalBVH;
import cl.core.data.struct.Bound;
import cl.core.data.struct.Node;
import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import coordinate.parser.OBJParser;
import org.jocl.CL;
import wrapper.core.OpenCLPlatform;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class Test2 {
    public static void main(String... args)
    {
        String plane =  "o Plane\n" +
                        "v -1.000000 0.000000 1.000000\n" +
                        "v 1.000000 0.000000 1.000000\n" +
                        "v -1.000000 0.000000 -1.000000\n" +
                        "v 1.000000 0.000000 -1.000000\n" +
                        "vn 0.0000 1.0000 0.0000\n" +
                        "s off\n" +
                        "f 2//1 3//1 1//1\n" +
                        "f 2//1 4//1 3//1";
        
        String directory = "raytracing/";       
        CL.setExceptionsEnabled(true);
        
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2 = CLFileReader.readFile(CLSource.class, "Primitive.cl");
        String source3 = CLFileReader.readFile(CLSource.class, "NormalBVH.cl");
        String source4 = CLFileReader.readFile(CLSource.class, "SimpleTrace.cl");
                
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(source1, source2, source3, source4);        
        
        CMesh mesh = new CMesh(configuration);
        OBJParser parser = new OBJParser();
        parser.read("C:\\Users\\user\\Documents\\Scene3d\\simplebox\\box.obj", mesh);
        
        CNormalBVH bvh = new CNormalBVH(configuration);
        bvh.build(mesh);
        
        CStructIntArray<Node> nodes = bvh.getNodes();
        CStructFloatArray<Bound> bounds = bvh.getBounds();
        
        for(int i = 0; i<bounds.getSize(); i++)
            System.out.println(bounds.get(i));
            
        
    }
}
