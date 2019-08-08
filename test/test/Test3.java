/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.shapes.CMesh;
import coordinate.parser.obj.OBJMappedParser;

/**
 *
 * @author user
 */
public class Test3 {
    public static void main(String... args)
    {
        OBJMappedParser parser = new OBJMappedParser();
        CMesh mesh = new CMesh(null);
        parser.read("C:\\Users\\user\\Documents\\Scene3d\\sphere\\sphere-cylcoords-1k.obj", mesh);
    }
}
