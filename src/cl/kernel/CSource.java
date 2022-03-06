/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.kernel;

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
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Util.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "SAT.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "RGBSpace.cl"));        
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Sampling.cl"));           
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Matrix.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Geometry.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Bsdf.cl"));        
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Primitive.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Material.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "NormalBVH.cl")); 
        stringBuilder.append(CLFileReader.readFile(CSource.class, "LightGrid.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Light.cl"));        
        stringBuilder.append(CLFileReader.readFile(CSource.class, "LightConfiguration.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Path.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Initialize.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Raytrace.cl"));
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Render.cl"));        
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Compact.cl"));
        
        stringBuilder.append(CLFileReader.readFile(CSource.class, "Image.cl"));
        return new String[]{stringBuilder.toString()};
    }
}
