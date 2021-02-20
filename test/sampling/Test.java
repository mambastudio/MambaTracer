/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling;

import static cl.abstracts.MambaAPIInterface.getGlobal;
import static cl.abstracts.MambaAPIInterface.getNumOfGroups;

/**
 *
 * @author user
 */
public class Test {
    public static void main(String... args){
        float[] array = new float[]{0, 1, 0, 1, 1, 1, 0, 0};
        Distribution1D dist = new Distribution1D(array, 0, array.length);
        
        float random = (float) Math.random();
        
        int[] offsetC = new int[1];
        float[] pdfC = new float[1];
        float[] pdfD = new float[1];
        
        float value = dist.sampleContinuous(random, pdfC, offsetC);
        int offsetD = dist.sampleDiscrete(random, pdfD);
        
        
//        System.out.println(pdfC[0]/array.length);
//        System.out.println(pdfD[0]);
//        
//        System.out.println((int)(value*array.length));
//        System.out.println(offsetD);
        
        int length = 250000;
        
        int local = 120;
        int global = getGlobal(length, local);
        int group = getNumOfGroups(global,local);
        
        System.out.println(global);
        System.out.println(group);
        System.out.println(global/local);
    }
}
