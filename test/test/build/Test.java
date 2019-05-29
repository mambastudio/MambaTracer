/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.build;

import cl.shapes.CMesh;
import coordinate.parser.OBJParser;
import java.math.BigInteger;
import java.util.Random;

/**
 *
 * @author user
 */
public class Test {
    int array[] = {};
    public static void main(String... args)
    {
        Random rnd = new Random();
        int seed = 729394943;//BigInteger.probablePrime(30, rnd).intValue();      
        System.out.println(seed); Math.random();
        System.out.println(Integer.bitCount(seed));
        System.out.println(rndFloat(seed));
    }
    
    public static float rndFloat(int seed)
    {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed * 2.3283064365387e-10f;
    }
}
