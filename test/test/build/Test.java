/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.build;

import cl.shapes.CMesh;
import coordinate.parser.OBJParser;

/**
 *
 * @author user
 */
public class Test {
    public static void main(String... args)
    {
        OBJParser parser = new OBJParser();
        CMesh mesh = new CMesh(null);
        parser.read("C:\\Users\\user\\Documents\\Scene3d\\planesphere.obj", mesh);
        GPUBuildBVH bvh = new GPUBuildBVH();
        bvh.build(mesh);
    }
}
