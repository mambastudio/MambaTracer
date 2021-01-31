/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling;

/**
 *
 * @author user
 */
public class Test {
    public static void main(String... args){
        float[] array = new float[]{0, 0, 0, 0, 1, 0, 0, 0};
        Distribution1D dist = new Distribution1D(array, 0, array.length);
        
        int[] offset = new int[1];
        float[] pdf = new float[1];
        float value = dist.sampleDiscrete((float) Math.random(), pdf);
        
        System.out.println(pdf[0]);
        System.out.println(dist.discretePDF(4));
    }
}
