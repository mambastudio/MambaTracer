/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import coordinate.struct.IntStruct;

/**
 *
 * @author user
 */
public class CFace extends IntStruct{
    public int     v1,  v2,  v3;
    public int     uv1, uv2, uv3;
    public int     n1,  n2,  n3;
    public int     mat;
    
    public CFace()
    {
        v1  = v2  = v3  = 0;
        uv1 = uv2 = uv3 = 0;
        n1  = n2  = n3  = 0;
        mat = -1;
    }
    
    
    @Override
    public void initFromGlobalArray() {
        int[] globalArray = getGlobalArray();
        if(globalArray == null)
            return;
        int globalArrayIndex = getGlobalArrayIndex();

        v1      = globalArray[globalArrayIndex + 0];
        v2      = globalArray[globalArrayIndex + 1];
        v3      = globalArray[globalArrayIndex + 2];
        uv1     = globalArray[globalArrayIndex + 3];
        uv2     = globalArray[globalArrayIndex + 4];
        uv3     = globalArray[globalArrayIndex + 5];
        n1      = globalArray[globalArrayIndex + 6];
        n2      = globalArray[globalArrayIndex + 7];
        n3      = globalArray[globalArrayIndex + 8];
        mat     = globalArray[globalArrayIndex + 9] & 0xFFFF;
    }

    @Override
    public int[] getArray() {
        return new int[]{v1, v2, v3, uv1, uv2, uv3, n1, n2, n3, mat};
    }

    @Override
    public int getSize() {
        return 10;
    }
    
    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder(); 
        builder.append("v1  ").append(v1).append(" v2  " ).append(v2).append(" v3  ").append(v3).append("\n");
        builder.append("uv1 ").append(uv1).append(" uv2 ").append(uv2).append(" uv3 ").append(uv3).append("\n");
        builder.append("n1 ").append(n1).append(" n2 ").append(n2).append(" n3 ").append(n3).append("\n");
        builder.append("mat ").append(mat);
         
        return builder.toString();
    }
}
