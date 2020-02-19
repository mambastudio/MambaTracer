/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import coordinate.generic.raytrace.AbstractAccelerator;
import java.util.concurrent.TimeUnit;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class SNormalBVH implements AbstractAccelerator
        <SRay, 
         SIsect,
         SMesh,
         SBound> {
    
     //Primitive
    SMesh primitives;
    int[] objects;
    
    //Tree, Primitive index, Boundingbox   
    CStructFloatArray<SBound> bounds;
    CStructIntArray<SNode> nodes;
    
    //Node counter
    int nodesPtr = 0;
    
    //Opencl configuration
    OpenCLPlatform configuration;
    
    public SNormalBVH(OpenCLPlatform configuration)
    {
        this.configuration = configuration;
    }
    
    @Override
    public void build(SMesh primitives) {
        long time1 = System.nanoTime();
        
        this.primitives = primitives; 
        this.objects = new int[primitives.getCount()];
        
        for(int i = 0; i<this.primitives.getCount(); i++)
            objects[i] = i;
        
        //Release memory
        CResourceFactory.releaseMemory("nodes", "bounds");
        
        //Allocate BVH root node
        nodes   = new CStructIntArray(configuration, SNode.class, this.primitives.getCount() * 2 - 1, "nodes", READ_WRITE);
        bounds  = new CStructFloatArray(configuration, SBound.class, this.primitives.getCount() * 2 - 1, "bounds", READ_WRITE);
        
        SNode root = new SNode();
        nodes.set(root, 0);        
        nodesPtr = 1;
                
        subdivide(0, 0, primitives.getCount());
        
        long time2 = System.nanoTime();
        
        System.out.println(nodes.getSize());
        
        long timeDuration = time2 - time1;
        String timeTaken= String.format("BVH build time: %02d min, %02d sec", 
                TimeUnit.NANOSECONDS.toMinutes(timeDuration), 
                TimeUnit.NANOSECONDS.toSeconds(timeDuration));
        System.out.println(timeTaken);    
        
        nodes.transferFromBufferToDevice();
        bounds.transferFromBufferToDevice();
    }
    private void subdivide(int parentIndex, int start, int end)
    {
        //Calculate the bounding box for the root node
        SBound bb = new SBound();
        SBound bc = new SBound();
        calculateBounds(start, end, bb, bc);
        
        nodes.index(parentIndex, parent -> parent.bound  = parentIndex);
        bounds.index(parentIndex, bound -> bound.setBound(bb));
            
        //Initialize leaf
        if(end - start < 2)
        {        
            nodes.index(parentIndex, (SNode parent) -> {
                parent.child = objects[start];
                parent.isLeaf = 1;
            });            
            return;
        }
        
         //Subdivide parent node
        SNode left;      
        SNode right;
int leftIndex, rightIndex;      
        synchronized(this)
        {
            left            = new SNode();   left.parent = parentIndex;
            right           = new SNode();   right.parent = parentIndex;
            
            nodes.set(left, nodesPtr);  leftIndex   = nodesPtr;   nodes.index(parentIndex, parent -> parent.left = nodesPtr++); 
            nodes.set(right, nodesPtr); rightIndex  = nodesPtr;   nodes.index(parentIndex, parent -> parent.right = nodesPtr++);  
            
            nodes.index(leftIndex, leftNode -> leftNode.sibling = rightIndex);
            nodes.index(rightIndex, rightNode -> rightNode.sibling = leftIndex);                      
        }   
        
        //set the split dimensions
        int split_dim = bc.maximumExtentAxis();        
        int mid = getMid(bc, split_dim, start, end);
                
        //Subdivide
        subdivide(leftIndex, start, mid);
        subdivide(rightIndex, mid, end);
    }
        
    private int getMid(SBound bc, int split_dim, int start, int end)
    {
        //split on the center of the longest axis
        float split_coord = bc.getCenter(split_dim);

        //partition the list of objects on this split            
        int mid = partition(primitives, objects, start, end, split_dim, split_coord);

        //if we get a bad split, just choose the center...
        if(mid == start || mid == end)
            mid = start + (end-start)/2;
        
        return mid;
    }
    
    private void calculateBounds(int first, int end, SBound bb, SBound bc)
    {                
        for(int p = first; p<end; p++)
        {
            bb.include(primitives.getBound(objects[p]));
            bc.include(primitives.getBound(objects[p]).getCenter());
        }        
    }

    @Override
    public boolean intersect(SRay ray, SIsect isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(SRay ray) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void intersect(SRay[] rays, SIsect[] isects) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public SBound getBound() {       
        return bounds.get(0).getCBound();
    }
    
    public CFloatBuffer getCBounds()
    {
        return bounds.getCBuffer();
    }
    
    public CStructFloatArray<SBound> getBounds()
    {
        return bounds;
    }
    
    public CIntBuffer getCNodes()
    {
        return nodes.getCBuffer();
    }
    
    public CStructIntArray<SNode> getNodes()
    {
        return nodes;
    }    
}
