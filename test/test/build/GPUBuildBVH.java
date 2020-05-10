/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.build;

import cl.core.CBoundingBox;
import cl.core.data.CPoint3;
import cl.core.data.struct.CBound;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CNode;
import cl.core.data.struct.CNodeRange;
import cl.core.data.struct.CRay;
import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import coordinate.generic.raytrace.AbstractAccelerator;
import coordinate.struct.FloatStruct;
import coordinate.struct.StructFloatArray;
import coordinate.struct.IntStruct;
import coordinate.struct.StructIntArray;
import org.jocl.CL;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class GPUBuildBVH implements AbstractAccelerator< CRay, 
                                                           CIntersection, 
                                                           CMesh, 
                                                           CBoundingBox> 
{
    CMesh           primitives;
     
    CStructFloatArray<CBound> bounds;
    CStructIntArray<CNode> nodes;
    CIntBuffer nodeCounter = null;
    
    CIntBuffer objects = null;
    CIntBuffer inputRangeCounter = null;
    CStructIntArray<CNodeRange> inputRange = null;   
    CIntBuffer outputRangeCounter = null;
    CStructIntArray<CNodeRange> outputRange = null;
    
    CIntBuffer counter = null;
    
    CIntBuffer swap = null;
    
    CKernel initBVHKernel;
    CKernel subdivideKernel;
    CKernel swapAndInitKernel;
    
    //Opencl configuration
    OpenCLPlatform configuration;
    
    public GPUBuildBVH(OpenCLPlatform configuration)
    {
        this.configuration = configuration;        
    }
    
    @Override
    public void build(CMesh primitives)
    {
                       
        this.primitives = primitives;
        this.primitives.initCLBuffers();
        
        //Release memory
        CResourceFactory.releaseMemory("nodes", "bounds");
        
        int primCount = this.primitives.getCount();
        
        //variables
        nodes   = new CStructIntArray(configuration, CNode.class, primCount * 2 - 1, "nodes", READ_WRITE);
        bounds  = new CStructFloatArray(configuration, CBound.class, primCount * 2 - 1, "bounds", READ_WRITE);
        nodeCounter = configuration.allocIntValue("nodecounter", 1, READ_WRITE);
        
        objects = configuration.allocInt("objects", primCount, READ_ONLY);
        inputRange = new CStructIntArray(configuration, CNodeRange.class, primCount, "inputranges", READ_WRITE);
        inputRangeCounter = configuration.allocIntValue("inputrangecounter", 0, READ_WRITE);
        outputRange = new CStructIntArray(configuration, CNodeRange.class, primCount, "outputranges", READ_WRITE);
        outputRangeCounter = configuration.allocIntValue("outputrangecounter", 0, READ_WRITE);
        
        swap = configuration.allocIntValue("swap", 0, READ_WRITE);
        
        counter = configuration.allocIntValue("counter", 1, READ_WRITE);
        
        //kernels
        initBVHKernel = configuration.createKernel("initBVH", objects, inputRange.getCBuffer(), inputRangeCounter, swap);
        subdivideKernel = configuration.createKernel("subdivide", primitives.clPoints(), primitives.clNormals(), primitives.clFaces(), primitives.clSize(),
                objects, inputRange.getCBuffer(), inputRangeCounter, outputRange.getCBuffer(), outputRangeCounter, nodes.getCBuffer(), bounds.getCBuffer(),
                nodeCounter, swap);
        swapAndInitKernel = configuration.createKernel("swapAndInit", inputRangeCounter, outputRangeCounter, counter, swap);
        
        long time1 = System.nanoTime();
        configuration.executeKernel1D(initBVHKernel, primCount, 1);
        
        while(counter.mapReadValue(configuration.queue()) > 0)
        {
            configuration.executeKernel1D(subdivideKernel, primCount, 1);
            configuration.executeKernel1D(swapAndInitKernel, 1, 1);
        }
        
        long time2 = System.nanoTime();
        
        
        System.out.println(primCount * 2 - 1);
        
        
        double mTime = (double)(time2 - time1)/1_000_000_000;
        System.out.printf("%.12f \n", mTime);
    }

    @Override
    public boolean intersect(CRay ray, CIntersection isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(CRay ray) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void intersect(CRay[] rays, CIntersection[] isects) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBoundingBox getBound() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
