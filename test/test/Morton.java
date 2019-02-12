/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import coordinate.struct.IntStruct;
import coordinate.struct.StructIntArray;

/**
 *
 * @author user
 */
public class Morton extends IntStruct
{
    public int mortonCode;
    public int primitiveIndex;

    @Override
    public void initFromGlobalArray() {
        int[] globalArray = getGlobalArray();
        if(globalArray == null)
            return;
        int globalArrayIndex = getGlobalArrayIndex();

        mortonCode      = globalArray[globalArrayIndex + 0];     
        primitiveIndex  = globalArray[globalArrayIndex + 1];                   
    }

    @Override
    public int[] getArray() {
        return new int[]{mortonCode, primitiveIndex};
    }

    @Override
    public int getSize() {
        return 2;
    }

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("primitive index - ").append(primitiveIndex).append(" ");
        builder.append("morton code     - ").append(mortonCode);
        return builder.toString();
    }
    
    public static void sort(StructIntArray<Morton> mortonPrimitives)
    {
        StructIntArray<Morton> temp = new StructIntArray<>(Morton.class, mortonPrimitives.size());
        int bitsPerPass = 6;
        int nBits = 30;
        int nPasses = nBits/bitsPerPass;
        
        for(int pass = 0; pass < nPasses; ++pass)
        {
            int lowBit = pass * bitsPerPass;
            
            StructIntArray<Morton> in  = ((pass & 1) == 1) ? temp              : mortonPrimitives;
            StructIntArray<Morton> out = ((pass & 1) == 1) ? mortonPrimitives  : temp;
            
            int nBuckets = 1 << bitsPerPass;
            int bucketCount[] = new int[nBuckets];
            int bitMask = (1 << bitsPerPass) - 1;
            
            for(int i = 0; i<in.size(); i++)
            {
                Morton p = in.get(i);
                int bucket = (p.mortonCode >> lowBit) & bitMask; 
                ++bucketCount[bucket];
            }
            
            int[] outIndex = new int[nBuckets];
            for(int i = 1; i < nBuckets; ++i)
                outIndex[i] = outIndex[i - 1] + bucketCount[i-1];
            
            for(int i = 0; i<in.size(); i++)
            {
                Morton p = in.get(i);
                int bucket = (p.mortonCode >> lowBit) & bitMask;
                out.set(p, outIndex[bucket]++);
            }            
        }
        
        if((nPasses & 1) == 1) System.arraycopy(temp.getArray(), 0, mortonPrimitives.getArray(), 0, temp.getArray().length);    
    }
}
