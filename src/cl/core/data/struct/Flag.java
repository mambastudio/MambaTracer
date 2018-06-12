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
public class Flag extends IntStruct {
    public int flag;

    @Override
    public void initFromGlobalArray() {
        int[] globalArray = getGlobalArray();
        if(globalArray == null)
            return;
        int globalArrayIndex = getGlobalArrayIndex();

        flag   = globalArray[globalArrayIndex + 0];       
    }

    @Override
    public int[] getArray() {
        return new int[]{flag};
    }

    @Override
    public int getSize() {
        return 1;
    } 

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder(); 
        builder.append("flag     ").append(flag).append("\n");        
        return builder.toString();
    }
}
