/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.accelerator;

import cl.core.ButterflySort;
import cl.core.CAccelerator;
import cl.core.CBoundingBox;
import cl.core.data.struct.CBound;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CNode;
import cl.core.data.struct.CRay;
import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import cl.shapes.CMesh;
import java.time.Duration;
import java.time.Instant;
import static wrapper.core.CBufferMemory.LOCALINT;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class CPlocBVH implements CAccelerator {
    
    int LOCALSIZE = 128;
    
    int leafS, nodeS, totalS;
    
    CMesh primitives;
    
    CStructIntArray<CNode> nodes;    
    CStructFloatArray<CBound> bounds;
    
    //Opencl configuration
    OpenCLPlatform configuration;
    
    public CPlocBVH(OpenCLPlatform configuration)
    {
        this.configuration = configuration;
    }

    @Override
    public void build(CMesh primitives) {
        this.primitives = primitives;
                
        leafS = primitives.getCount();
        nodeS = leafS - 1;
        totalS = nodeS + leafS;
        
        //variables
        nodes     = new CStructIntArray(configuration, CNode.class, totalS, "nodes", READ_WRITE);
        bounds = new CStructFloatArray(configuration, CBound.class, totalS, "bounds", READ_WRITE);
        
        CStructIntArray<CNode> output = new CStructIntArray(configuration, CNode.class, leafS, "output", READ_WRITE);
        CStructFloatArray<CBound> outputbounds = new CStructFloatArray(configuration, CBound.class, leafS, "outputbounds", READ_WRITE);
        CStructIntArray<CNode> input = new CStructIntArray(configuration, CNode.class, leafS, "input", READ_WRITE);
        CStructFloatArray<CBound> inputbounds = new CStructFloatArray(configuration, CBound.class, leafS, "inputbounds", READ_WRITE);
    
        CIntBuffer nearest = configuration.allocInt("nearest", leafS, READ_WRITE);
        CIntBuffer end  = configuration.allocIntValue("end", leafS, READ_WRITE);
        CIntBuffer radius = configuration.allocIntValue("radius", 500, READ_WRITE);
        CIntBuffer node_out_idx = configuration.allocIntValue("node_out_idx", leafS, READ_WRITE);
        
        CIntBuffer predicate        = configuration.allocInt("predicate", leafS, READ_WRITE);
        CIntBuffer localscan        = configuration.allocInt("localscan", leafS, READ_WRITE);
        CIntBuffer groupsum         = configuration.allocInt("groupsum" , getNumOfGroups(leafS, LOCALSIZE), READ_WRITE);
        CIntBuffer groupprefixsum   = configuration.allocInt("groupprefixsum", getNumOfGroups(leafS, LOCALSIZE), READ_WRITE);
        CIntBuffer groupsize        = configuration.allocIntValue("groupsize", getNumOfGroups(leafS, LOCALSIZE), READ_WRITE);
        CIntBuffer localsize        = configuration.allocIntValue("localsize", LOCALSIZE, READ_WRITE);
        CIntBuffer compactlength    = configuration.allocIntValue("compactlength", leafS, READ_WRITE);
    
        ButterflySort bsort = new ButterflySort(configuration, nodes, leafS);
               
        //kernels
        CKernel prepareMortonKernel         = configuration.createKernel("prepareMorton", primitives.clPoints(), primitives.clNormals(), primitives.clFaces(), primitives.clSize(), nodes.getCBuffer());
        CKernel prepareSortedLeafsKernel    = configuration.createKernel("prepareSortedLeafs", primitives.clPoints(), primitives.clNormals(), primitives.clFaces(), primitives.clSize(), nodes.getCBuffer(), 
                                                                                    bounds.getCBuffer(), input.getCBuffer(), inputbounds.getCBuffer());
        CKernel nearestKernel               = configuration.createKernel("nearest", inputbounds.getCBuffer(), outputbounds.getCBuffer(), nearest, end, radius);
        CKernel mergeKernel                 = configuration.createKernel("merge", input.getCBuffer(), inputbounds.getCBuffer(), output.getCBuffer(), outputbounds.getCBuffer(), nodes.getCBuffer(), bounds.getCBuffer(),
                                                                                    nearest, end, node_out_idx, predicate, localscan, groupsum, LOCALINT);
        CKernel prepareNextKernel           = configuration.createKernel("prepareNext", end, compactlength);
        CKernel groupPrefixSumKernel        = configuration.createKernel("groupPrefixSum", predicate, groupsum, groupprefixsum, localscan, groupsize, localsize, compactlength);
        CKernel compactKernel               = configuration.createKernel("compact", predicate, localscan, groupprefixsum, output.getCBuffer(), outputbounds.getCBuffer(), input.getCBuffer(), inputbounds.getCBuffer(), end);
    
        Instant start = Instant.now();
        //execute kernels
        configuration.executeKernel1D(prepareMortonKernel, getGlobal(leafS, LOCALSIZE), LOCALSIZE);
        bsort.sort(); //sort once
        configuration.executeKernel1D(prepareSortedLeafsKernel, getGlobal(leafS, LOCALSIZE), LOCALSIZE);
        
        //loop 
        while(end.mapReadValue(configuration.queue()) > 1)
        {
            //System.out.println("kubafu");
            configuration.executeKernel1D(nearestKernel, getGlobal(leafS, LOCALSIZE), LOCALSIZE);
            configuration.executeKernel1D(mergeKernel, getGlobal(leafS, LOCALSIZE), LOCALSIZE);
            configuration.executeKernel1D(groupPrefixSumKernel, 1, 1);
            configuration.executeKernel1D(compactKernel, getGlobal(leafS, LOCALSIZE), LOCALSIZE);
            configuration.executeKernel1D(prepareNextKernel, 1, 1);
            configuration.queue().finish();          
           
        }
        
        Instant stop = Instant.now();
        System.out.println("Ploc BVH Build " +Duration.between(start, stop).toMillis()/1000f +" seconds");
        
        bounds.transferFromDeviceToBuffer();
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
        return bounds.get(totalS-1).getCBound();
    }
    
    @Override
    public CFloatBuffer getCBounds()
    {
        return bounds.getCBuffer();
    }
    
    @Override
    public CStructFloatArray<CBound> getBounds()
    {
        return bounds;
    }
    
    @Override
    public CIntBuffer getCNodes()
    {
        return nodes.getCBuffer();
    }
    
    @Override
    public CStructIntArray<CNode> getNodes()
    {
        return nodes;
    }    
    
    public int getGlobal(int size, int LOCALSIZE)
    {
        if (size % LOCALSIZE == 0) { 
            return (int) ((Math.floor(size / LOCALSIZE)) * LOCALSIZE); 
        } else { 
            return (int) ((Math.floor(size / LOCALSIZE)) * LOCALSIZE) + LOCALSIZE; 
        } 
    }
    
    public int getNumOfGroups(int length, int LOCALSIZE)
    {
        int a = length/LOCALSIZE;
        int b = length%LOCALSIZE; //has remainder
        
        return (b > 0)? a + 1 : a;
            
    }
    
    @Override
    public int getStartNodeIndex()
    {
        return totalS-1;
    }
    
}
