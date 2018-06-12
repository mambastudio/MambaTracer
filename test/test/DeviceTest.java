/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.data.CPoint3;
import coordinate.struct.FloatStruct;
import coordinate.struct.StructFloatArray;
import org.jocl.CL;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class DeviceTest {
    public static void main(String... args)
    {       
        CL.setExceptionsEnabled(true);
        
        String directory = "C:\\Users\\user\\Documents\\OpenCL";
        String source1 = CLFileReader.readFile(directory, "Common.cl");
        String source2 = CLFileReader.readFile(directory, "DeviceTest.cl");        
        OpenCLPlatform configuration = OpenCLPlatform.getDefault(source1, source2);
        
        StructFloatArray nodeArray = new StructFloatArray(Node.class, 1);
        Node node = new Node();
        node.minimum.x = 21;
        node.maximum.z = 32;
        nodeArray.set(node, 0);
        CFloatBuffer nodeBuffer = configuration.createFromFloatArray("Node", READ_ONLY, nodeArray.getArray());
        
        CKernel kernel = configuration.createKernel("test", nodeBuffer);        
        configuration.executeKernel1D(kernel, 1, 1);
        
        CResourceFactory.releaseAll();
        
    }
    
    public static class Node extends FloatStruct
    {
        public CPoint3 minimum;
        public CPoint3 maximum;
        
        public Node()
        {
            minimum = new CPoint3();
            maximum = new CPoint3();
        }
        
        @Override
        public void initFromGlobalArray() {
            float[] globalArray = getGlobalArray();
            if(globalArray == null)
                return;
            int globalArrayIndex = getGlobalArrayIndex();
            
            minimum.x = globalArray[globalArrayIndex + 0];
            minimum.y = globalArray[globalArrayIndex + 1];
            minimum.z = globalArray[globalArrayIndex + 2];
            maximum.x = globalArray[globalArrayIndex + 4];
            maximum.y = globalArray[globalArrayIndex + 5];
            maximum.z = globalArray[globalArrayIndex + 6];
        }

        @Override
        public float[] getArray() {
            return new float[]{minimum.x, minimum.y, minimum.z, 0, maximum.x, maximum.y, maximum.z, 0};
        }

        @Override
        public int getSize() {
            return 8;
        }
        
    }
}
