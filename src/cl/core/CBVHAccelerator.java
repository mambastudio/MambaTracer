/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.shapes.CMesh;
import coordinate.generic.raytrace.AbstractAccelerator;
import java.util.concurrent.TimeUnit;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CBVHAccelerator implements AbstractAccelerator
        <CRay, 
         CIntersection, 
         CMesh, 
         CBoundingBox> {
    
     //Primitive
    CMesh primitives;
    
    //Tree, Primitive index, Boundingbox
    int[] objects;
    CBVHNode[] nodes = null;
    CBoundingBox bound = null;
    
    //node counter
    int nodeCount = 0;

    @Override
    public void build(CMesh primitives) {
        long time1 = System.nanoTime();
        
        this.primitives = primitives;  
        objects = new int[this.primitives.getCount()];
        for(int i = 0; i<this.primitives.getCount(); i++)
            objects[i] = i;
        bound = this.primitives.getBound();
        
        //Allocate BVH root node
        nodes = new CBVHNode[this.primitives.getCount() * 2 - 1];
        CBVHNode root = new CBVHNode();
        nodes[0] = root;
        nodeCount = 1;
        
        subdivide(root, 0, objects.length);
        
        long time2 = System.nanoTime();
        
        System.out.println(nodes.length);
        
        long timeDuration = time2 - time1;
        String timeTaken= String.format("BVH build time: %02d min, %02d sec", 
                TimeUnit.NANOSECONDS.toMinutes(timeDuration), 
                TimeUnit.NANOSECONDS.toSeconds(timeDuration));
        System.out.println(timeTaken);        
    }
    
    private void subdivide(CBVHNode parent, int start, int end)
    {
        //Calculate the bounding box for the root node
        CBoundingBox bb = new CBoundingBox();
        CBoundingBox bc = new CBoundingBox();
        calculateBounds(start, end, bb, bc);
        parent.bounds = bb;
                
        //Initialize leaf
        if(end - start == 1)
        {            
            parent.primOffset = start;                  
            return;
        }
        
        //Subdivide parent node        
        CBVHNode left, right;        
        synchronized(this)
        {
            left            = new CBVHNode();
            right           = new CBVHNode();                   
        }   
        
        //set the split dimensions
        int split_dim = bc.maximumExtent();        
        int mid = getMid(bc, split_dim, start, end);
                
        //Subdivide
        nodes[nodeCount++] = left;  
        subdivide(left, start, mid); 
        
        parent.skipIndex = nodeCount; //PLEASE NOTE HERE
        
        nodes[nodeCount++] = right; 
        subdivide(right, mid, end);  
    }
    
    private int getMid(CBoundingBox bc, int split_dim, int start, int end)
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
    
    private void calculateBounds(int first, int end, CBoundingBox bb, CBoundingBox bc)
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
    public CBoundingBox getBound() {
        return bound;
    }
    
    public int[] getObjectIDs()
    {
        return objects;
    }
    
    public CBVHNode[] getBVHNodes()
    {
        return nodes;
    }
    
    public int getBVHNodesSize()
    {
        return nodes.length;
    }
        
    public static class CBVHNode extends Struct
    {
        public CBoundingBox bounds;
        public int primOffset, skipIndex;        
    }
    
}
