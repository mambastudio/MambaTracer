/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package raytrace.cl;

import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class RaytraceSource {
    public static String[] readFiles()
    {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Print.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Util.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Geometry.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Sampling.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "RGBSpace.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Matrix.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Primitive.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "NormalBVH.cl"));        
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Bsdf.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Material.cl"));
        stringBuilder.append(CLFileReader.readFile(RaytraceSource.class, "Raytrace.cl"));
        return new String[]{stringBuilder.toString()};
    }
}
