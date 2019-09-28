/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

/**
 *
 * @author user
 */
public class Test3 {
    
    private static final int LOCALSIZECONSTANT = 8;
    
    public static void main(String... args)
    {
        //OBJMappedParser parser = new OBJMappedParser();
        //CMesh mesh = new CMesh(null);
        //parser.read("C:\\Users\\user\\Documents\\Scene3d\\sphere\\sphere-cylcoords-1k.obj", mesh);
        
        int length = 599;
        System.out.println(length1(length));
        System.out.println(length2(length));
        System.out.println(length3(length));
        System.out.println(length(length, 1));
        System.out.println(length(length, 2));
        System.out.println(length(length, 3));
    }
    
    public static int pow2length(int length)
    {
        int log2 = log2(length);
        int difference = (int)(Math.pow(2, log2)) - length;
        
        if(difference == 0) return length;
        else                return (int) Math.pow(2, log2+1);
    }
    
    public static int log2( int bits ) // returns 0 for bits=0
    {
        int log = 0;
        if( ( bits & 0xffff0000 ) != 0 ) { bits >>>= 16; log = 16; }
        if( bits >= 256 ) { bits >>>= 8; log += 8; }
        if( bits >= 16  ) { bits >>>= 4; log += 4; }
        if( bits >= 4   ) { bits >>>= 2; log += 2; }
        return log + ( bits >>> 1 );
    }
    
    public static int length1(int size)
    {
        int length = pow2length(size);
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 3);
        if(full_length == 0)
            return 1;
        else if(length > full_length)
            return full_length;
        else
            return length;
    }
    
    public static int length2(int size)
    {
        int length = length1(size); length /= LOCALSIZECONSTANT;
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 2);
        if(length > full_length)
            return full_length;
        else
            return length;
    }
    
    public static int length3(int size)
    {
        int length = length2(size); length /= LOCALSIZECONSTANT;
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, 1);
        if(length > full_length)
            return full_length;
        else
            return length;
    }    
    
    public static int length(int size, int level)
    {
        int length = pow2length(size);
        
        length /= (int)Math.pow(LOCALSIZECONSTANT, level - 1);
        
        if(length == 0)
            return 1;
        int full_length = (int) Math.pow(LOCALSIZECONSTANT, log2(length));
        if(full_length == 0)
            return 1;
        else if(length > full_length)
            return full_length;
        else
            return length;
    }
}
