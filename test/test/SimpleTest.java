/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.data.struct.CIntersection;
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
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class SimpleTest {
    public static void main(String... args)
    {
        CPlatform platform = CConfiguration.getDefault();
        CDevice device = platform.getDefaultDevice();        
        CContext context = platform.createContext(device);
        CProgram program = context.createProgram(programSource);        
        CCommandQueue queue = context.createCommandQueue(device);
        
        // size
        int n = 10;
                
        //write input
        CStructTypeBuffer<CIntersection> intersectionBuffer = CBufferFactory.allocStructType("intersections", context, CIntersection.class, n, READ_WRITE);
        intersectionBuffer.mapWriteBuffer(queue, intersections->{
            for(CIntersection intersection: intersections)
                intersection.setMat(4);
        });
        
         //execute kernel
        CKernel kernel = program.createKernel("test");
        kernel.putArgs(intersectionBuffer);        
        queue.put1DRangeKernel(kernel, n, 1); 
        
        //read output
        intersectionBuffer.mapReadBuffer(queue, intersections-> {
            for(CIntersection intersection : intersections)
            {
                System.out.println(intersection.mat);
                System.out.println(intersection.pixel);
                System.out.println(intersection.throughput);
                
            }
            
        });        
        
        CResourceFactory.releaseAll();
    }
    
    private static final String programSource =
        
        // Definition of the Particle struct in the kernel
        "typedef struct\n" +
        "{\n" +        
        "   float4 throughput;"+    
        "   float4 p;"+    
        "   float4 n;"+    
        "   float4 d;"+    
        "   float2 pixel;"+ 
        "   float2 uv;"+ 
        "   int mat;" + 
        "   int sampled_brdf;" + 
        "   int id;" + 
        "   int hit;" + 
        "}Intersection;"+

        // The actual kernel, performing some dummy computation
        "__kernel void test(__global Intersection* intersections)"+ "\n" +
        "{"+ "\n" +
        "    int gid = get_global_id(0);"+ "\n" +          
        "    intersections[gid].throughput  =  154;"+ "\n" +
        "    intersections[gid].pixel  =  1;"+ "\n" +
        "    intersections[gid].hit  =  gid;"+ "\n" +   
        "    intersections[gid].mat  *=  gid;"+ "\n" +
        "}";
}
