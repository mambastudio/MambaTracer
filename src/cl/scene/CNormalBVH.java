/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.scene;

import cl.abstracts.CAcceleratorInterface;
import cl.struct.CNode;
import cl.struct.CIntersection;
import cl.struct.CRay;
import cl.struct.CBound;
import coordinate.generic.raytrace.AbstractAccelerator;
import java.util.concurrent.TimeUnit;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLConfiguration;

/**
 *
 * @author user
 */
public class CNormalBVH implements CAcceleratorInterface {
    
     //Primitive
    CMesh primitives;
    int[] objects;
    
    //Tree, Primitive index, Boundingbox   
    CMemory<CBound> bounds;
    CMemory<CNode> nodes;
    
    //Node counter
    int nodesPtr = 0;
    
    //Opencl configuration
    OpenCLConfiguration configuration;
    
    public CNormalBVH(OpenCLConfiguration configuration)
    {
        this.configuration = configuration;
    }
    
    @Override
    public void build(CMesh primitives) {
        long time1 = System.nanoTime();
        
        this.primitives = primitives; 
        this.objects = new int[primitives.getCount()];
        
        for(int i = 0; i<this.primitives.getCount(); i++)
            objects[i] = i;
        
        //Release memory
        CResourceFactory.releaseMemory("nodes", "bounds");
        
        //Allocate BVH root node
        nodes   = configuration.createBufferI(CNode.class, this.primitives.getCount() * 2 - 1,  READ_WRITE);
        bounds  = configuration.createBufferF(CBound.class, this.primitives.getCount() * 2 - 1,  READ_WRITE);
        
        CNode root = new CNode();
        nodes.setCL(root);        
        nodesPtr = 1;
                
        subdivide(0, 0, primitives.getCount());
        
        long time2 = System.nanoTime();
        
        System.out.println(nodes.getSize());
        
        long timeDuration = time2 - time1;
        String timeTaken= String.format("BVH build time: %02d min, %02d sec", 
                TimeUnit.NANOSECONDS.toMinutes(timeDuration), 
                TimeUnit.NANOSECONDS.toSeconds(timeDuration));
        System.out.println(timeTaken);    
        
        nodes.transferToDevice();
        bounds.transferToDevice();
    }
    private void subdivide(int parentIndex, int start, int end)
    {
        //Calculate the bounding box for the root node
        CBound bb = new CBound();
        CBound bc = new CBound();
        calculateBounds(start, end, bb, bc);
        
        nodes.index(parentIndex, parent -> parent.bound  = parentIndex);
        bounds.index(parentIndex, bound -> bound.setBound(bb));
            
        //Initialize leaf
        if(end - start < 2)
        {        
            nodes.index(parentIndex, (CNode parent) -> {
                parent.child = objects[start];
                parent.isLeaf = 1;
            });            
            return;
        }
        
         //Subdivide parent node
        CNode left;      
        CNode right;
        int leftIndex, rightIndex;      
        synchronized(this)
        {
            left            = new CNode();   left.parent = parentIndex;
            right           = new CNode();   right.parent = parentIndex;
            
            nodes.set(nodesPtr, left); leftIndex   = nodesPtr;   nodes.index(parentIndex, parent -> parent.left = nodesPtr++); 
            nodes.set(nodesPtr, right); rightIndex  = nodesPtr;   nodes.index(parentIndex, parent -> parent.right = nodesPtr++);  
            
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
        
    private int getMid(CBound bc, int split_dim, int start, int end)
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
    
    private void calculateBounds(int first, int end, CBound bb, CBound bc)
    {                
        for(int p = first; p<end; p++)
        {
            bb.include(primitives.getBound(objects[p]));
            bc.include(primitives.getBound(objects[p]).getCenter());
        }        
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
    public CBound getBound() {       
        return bounds.get(0).getCopy();
    }

    @Override
    public CMemory<CBound> getBounds() {
        return bounds;
    }

    @Override
    public CMemory<CNode> getNodes() {
        return nodes;
    }
    
}
