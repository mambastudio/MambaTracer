/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.material;

import coordinate.struct.refl.ByteStructInfo;
import java.util.Arrays;
import org.jocl.struct.Struct;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CStructTypeBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class TestMaterial {
    public static void main(String... args)
    {       
        /*
        Struct.showLayout(JMaterial.class);
        ByteStructInfo info = new ByteStructInfo(CMaterial.class);
        System.out.println(Arrays.toString(info.offsets()));
        
        System.out.println(computeAlignmentOffset(48, 4));
        */
        
        
        int global                      = 5;
        String source                   = CLFileReader.readFile(TestMaterial.class, "TestMaterial.cl");
        OpenCLPlatform configuration    = OpenCLPlatform.getDefault(source);
        
        CStructTypeBuffer<CMaterial> materials = CBufferFactory.allocStructType("materials", configuration.context(), CMaterial.class, global, READ_WRITE);
        materials.mapWriteBuffer(configuration.queue(), cmats -> {
            CMaterial mat = cmats.get(1);
            mat.setEmitterEnabled(true); 
            System.out.println(mat.getMaterial());
        });
        
        System.out.println("executing... ");
        
        CKernel kernel = configuration.createKernel("test", materials);        
        configuration.executeKernel1D(kernel, global, 1);
        
        materials.mapReadBuffer(configuration.queue(), cmats-> {
            CMaterial mat = cmats.get(1);
            System.out.println(mat.getMaterial());
        });
        CResourceFactory.releaseAll();

    }    
}
