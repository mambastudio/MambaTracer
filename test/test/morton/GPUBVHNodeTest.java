/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.morton;

import cl.core.data.CPoint3;
import cl.core.kernel.CLSource;
import coordinate.struct.FloatStruct;
import coordinate.struct.StructFloatArray;
import coordinate.struct.IntStruct;
import coordinate.struct.StructIntArray;
import org.jocl.CL;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class GPUBVHNodeTest {
    public static void main(String... args)
    {
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");        
        String source2 = CLFileReader.readFile(GPUBVHNodeTest.class, "GPUBVHNode.cl");
        
        CL.setExceptionsEnabled(true);
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(source1, source2);
          
        int leafS = 5;              // n
        int nodeS = leafS - 1;      // n - 1 
        int tSize = nodeS + leafS;  //2n - 1
        
        StructFloatArray<BoundingBox> bounds = new StructFloatArray<>(BoundingBox.class, tSize);        
        StructIntArray<BVHNode>       nodes  = new StructIntArray<>(BVHNode.class, tSize); 
                
        CIntBuffer      cnodes          = CBufferFactory.wrapInt("nodes", configuration.context(), configuration.queue(), nodes.getArray(), READ_WRITE);
        CFloatBuffer    cbounds         = CBufferFactory.wrapFloat("bounds", configuration.context(), configuration.queue(), bounds.getArray(), READ_WRITE);
        CIntBuffer      cnodeSize       = CBufferFactory.initIntValue("nodeSize", configuration.context(), configuration.queue(), nodeS, READ_ONLY);
        CIntBuffer      cleafSize       = CBufferFactory.initIntValue("leafSize", configuration.context(), configuration.queue(), leafS, READ_ONLY);
        
        
        CKernel testKernel = configuration.program().createKernel("test",cnodes, cbounds, cnodeSize, cleafSize); 
        configuration.queue().put1DRangeKernel(testKernel, 1, 1);
        
        cnodes.transferFromDeviceToBuffer(configuration.queue());
        cbounds.transferFromDeviceToBuffer(configuration.queue());
       
        for(int i = 0; i<leafS; i++)
            System.out.println(nodes.get(i));
    }
    
    public static class BVHNode extends IntStruct
    {
        int bound;
        int sibling;
        int left;
        int right;
        int parent;
        int isLeaf;
        int child;
        
        @Override
        public void initFromGlobalArray() {
            int[] globalArray = getGlobalArray();
            if(globalArray == null)
                return;
            int globalArrayIndex = getGlobalArrayIndex();
            
            bound   = globalArray[globalArrayIndex + 0];
            sibling = globalArray[globalArrayIndex + 1];
            left    = globalArray[globalArrayIndex + 2];
            right   = globalArray[globalArrayIndex + 3];
            parent  = globalArray[globalArrayIndex + 4];
            isLeaf  = globalArray[globalArrayIndex + 5];
            child   = globalArray[globalArrayIndex + 6];
        }

        @Override
        public int[] getArray() {
            return new int[]{bound, sibling, left, right, parent, isLeaf, child};
        }

        @Override
        public int getSize() {
            return 7;
        } 
        
        @Override
        public String toString()
        {
            StringBuilder builder = new StringBuilder(); 
            builder.append("bounds   ").append(bound).append("\n");
            builder.append("parent   ").append(parent).append("\n");
            builder.append("sibling  ").append(sibling).append("\n");
            builder.append("left     ").append(left).append(" right     ").append(right).append("\n");
            builder.append("is leaf  ").append(isLeaf).append("\n");
            builder.append("child no ").append(child).append("\n");    
            return builder.toString();
        }
    }
    
public static class BoundingBox extends FloatStruct
{
    CPoint3 minimum;
    CPoint3 maximum;

    public BoundingBox()
    {
        this.minimum = new CPoint3();
        this.maximum = new CPoint3();
    }

    @Override
    public void initFromGlobalArray() {
        float[] globalArray = getGlobalArray();
        if(globalArray == null)
            return;
        int globalArrayIndex = getGlobalArrayIndex();

        minimum.x   = globalArray[globalArrayIndex + 0];
        minimum.y   = globalArray[globalArrayIndex + 1];
        minimum.z   = globalArray[globalArrayIndex + 2];
        maximum.x   = globalArray[globalArrayIndex + 4];
        maximum.y   = globalArray[globalArrayIndex + 5];
        maximum.z   = globalArray[globalArrayIndex + 6];
    }

    @Override
    public float[] getArray() {
        return new float[]{minimum.x, minimum.y, minimum.z, 0,
                           maximum.x, maximum.y, maximum.z, 0};
    }

    @Override
    public int getSize() {
        return 8;
    } 

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder(); 
        builder.append("Bounding Box").append("\n");
        builder.append("  ").append("minimum: ").append(minimum).append("\n");
        builder.append("  ").append("maximum: ").append(maximum).append("\n");
        return builder.toString();
    }
}    
}
