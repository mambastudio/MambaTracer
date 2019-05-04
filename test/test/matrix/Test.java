/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.matrix;

import cl.core.CCamera;
import cl.core.CCamera.CameraStruct;
import cl.core.CTransform;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.kernel.CLSource;
import org.jocl.struct.Struct;
import wrapper.core.CBufferFactory;
import wrapper.core.CCommandQueue;
import wrapper.core.CConfiguration;
import wrapper.core.CContext;
import wrapper.core.CDevice;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CPlatform;
import wrapper.core.CProgram;
import wrapper.core.CResourceFactory;
import wrapper.core.buffer.CStructBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class Test {
    public static void main(String... args)
    {
        camTest();
    }
    
    public static void camTest()
    {
        CPlatform platform = CConfiguration.getDefault();
        CDevice device = platform.getDefaultDevice();        
        CContext context = platform.createContext(device);
                    
        String source1 = CLFileReader.readFile(CLSource.class, "Common.cl");
        String source2 = CLFileReader.readFile(Test.class, "TestMatrix.cl");
        
        CProgram program = context.createProgram(source1, source2);
        CCommandQueue queue = context.createCommandQueue(device);
        
        int globalSize = 1;
        
        CCamera camera = new CCamera(new CPoint3(7, 89, 9), new CPoint3(2, 1, 15), new CVector3(2, 1, 0), 45); 
        camera.setUp();
        
        CStructBuffer<CameraStruct> transformBuffer = CBufferFactory.allocStruct("particles", context, CameraStruct.class, globalSize, READ_WRITE);
        transformBuffer.mapWriteBuffer(queue, buffer -> {
            buffer[0] = camera.getCameraStruct();
        });
        
        //execute kernel
        CKernel kernel = program.createKernel("camTest");
        kernel.putArgs(transformBuffer);        
        queue.put1DRangeKernel(kernel, globalSize, 1);   
        
        CResourceFactory.releaseAll();
        
        camera.setUp();
        CPoint3 p = new CPoint3();
        camera.cameraTransform.inverse().transformAssign(p);        
       
    }     
}
