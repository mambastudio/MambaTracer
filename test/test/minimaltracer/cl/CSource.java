/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer.cl;

import wrapper.core.algorithms.CPrefixSum;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class CSource {
    public static String[] readFiles()
    {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Print.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Sampling.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Util.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Matrix.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Primitive.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Material.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "NormalBVH.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Light.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Raytrace.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Render.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Compact.cl"));
        return new String[]{stringBuilder.toString()};
    }
}
