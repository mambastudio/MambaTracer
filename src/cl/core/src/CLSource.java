/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.src;

import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class CLSource {
    public static String[] readFiles()
    {
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2 = CLFileReader.readFile(CLSource.class, "Primitive.cl");
        String source3 = CLFileReader.readFile(CLSource.class, "NormalBVH.cl");
        String source4 = CLFileReader.readFile(CLSource.class, "SimpleTrace.cl");
        return new String[]{source1, source2, source3, source4} ;
    }
}
