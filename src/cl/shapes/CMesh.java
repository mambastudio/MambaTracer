/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.shapes;

import cl.core.CBVHAccelerator;
import cl.core.CBVHAccelerator.CBVHNode;
import cl.core.CBoundingBox;
import cl.core.CIntersection;
import cl.core.CRay;
import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.generic.AbstractMesh;
import coordinate.generic.raytrace.AbstractPrimitive;
import coordinate.list.CoordinateList;
import coordinate.list.IntList;
import wrapper.core.CBufferFactory;
import wrapper.core.CCommandQueue;
import wrapper.core.CContext;
import static wrapper.core.CMemory.READ_ONLY;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;
import wrapper.core.svm.CSVMFloatBuffer;
import wrapper.core.svm.CSVMIntBuffer;

/**
 *
 * @author user
 */
public class CMesh extends AbstractMesh<CPoint3, CVector3, CPoint2> implements AbstractPrimitive
        <CPoint3,         
         CRay, 
         CIntersection, 
         CBVHAccelerator, 
         CBoundingBox> {
    
    CBVHAccelerator accelerator;
    CBoundingBox bounds;
    
    public CMesh()
    {
        points = new CoordinateList(CPoint3.class);
        normals = new CoordinateList(CVector3.class);
        texcoords = new CoordinateList(CPoint2.class);
        triangleFaces = new IntList();
        bounds = new CBoundingBox();
    }

    @Override
    public int getCount() {
        return triangleSize();
    }

    @Override
    public CBoundingBox getBound(int primID) {
        CBoundingBox bbox = new CBoundingBox();
        bbox.include(getVertex1(primID));
        bbox.include(getVertex2(primID));
        bbox.include(getVertex3(primID));
        return bbox; 
    }

    @Override
    public CPoint3 getCentroid(int primID) {
        return getBound(primID).getCenter();
    }

    @Override
    public CBoundingBox getBound() {
        return bounds;
    }

    @Override
    public boolean intersect(CRay r, int primID, CIntersection isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(CRay r, int primID) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersect(CRay r, CIntersection isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(CRay r) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBVHAccelerator getAccelerator() {
        return accelerator;
    }

    @Override
    public void buildAccelerator() {
        this.accelerator = new CBVHAccelerator();
        this.accelerator.build(this);
    }

    @Override
    public float getArea(int primID) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void addPoint(CPoint3 p) {
        points.add(p);
        bounds.include(p);
    }

    @Override
    public void addPoint(float... values) {
        CPoint3 p = new CPoint3(values[0], values[1], values[2]);
        points.add(p);
        bounds.include(p);
    }

    @Override
    public void addNormal(CVector3 n) {
        normals.add(n);
    }

    @Override
    public void addNormal(float... values) {
        CVector3 n = new CVector3(values[0], values[1], values[2]);
        normals.add(n);
    }

    @Override
    public void addTexCoord(CPoint2 t) {
        texcoords.add(t);
    }

    @Override
    public void addTexCoord(float... values) {
        CPoint2 p = new CPoint2(values[0], values[1]);
        texcoords.add(p);
    }
    
    public CIntBuffer getCLObjectIdBuffer(String name, CContext context, CCommandQueue queue)
    {
        CIntBuffer buffer = CBufferFactory.allocInt(name, context, accelerator.getObjectIDs().length, READ_ONLY);
        buffer.mapWriteBuffer(queue, intbuffer -> intbuffer.put(accelerator.getObjectIDs()));        
        return buffer;
    }
    
    public CStructBuffer<CBVHNode> getCLBVHNodeArray(String name, CContext context, CCommandQueue queue)
    {
        CStructBuffer<CBVHNode> buffer = CBufferFactory.allocStruct(name, context, CBVHNode.class, accelerator.getBVHNodes().length, READ_ONLY);
        buffer.mapWriteBuffer(queue, array -> 
        {
            CBVHNode[] nodes = accelerator.getBVHNodes();
            for(int i = 0; i<nodes.length; i++)
            {
                if(nodes[i] != null)
                    array[i] = nodes[i];
            }            
        });
        return buffer;
    }
    
    public CIntBuffer getCLBVHNodeArraySize(String name, CContext context, CCommandQueue queue)
    {
        CIntBuffer buffer = CBufferFactory.allocInt(name, context, 1, READ_ONLY);
        buffer.mapWriteBuffer(queue, intbuffer -> intbuffer.put(accelerator.getBVHNodesSize()));        
        return buffer;
    }
    
    public CFloatBuffer getCLPointsBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapFloat(name, context, queue, getPointArray(), READ_ONLY);          
    }
    
    public CIntBuffer getCLFacesBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapInt(name, context, queue, getTriangleFacesArray(), READ_ONLY);
    }
    
    public CIntBuffer getCLSizeBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapInt(name, context, queue, new int[]{triangleSize()}, READ_ONLY);               
    }   
    
    public CSVMFloatBuffer getCLPointsBufferSVM(CContext context, CCommandQueue queue)
    {
        CSVMFloatBuffer buffer = CBufferFactory.allocSVMFloat(context, pointArraySize(), READ_ONLY);   
        buffer.mapWriteBuffer(queue, floatbuffer -> floatbuffer.put(getPointArray()));        
        return buffer;
    }
    
    public CSVMIntBuffer getCLFacesBufferSVM(CContext context, CCommandQueue queue)
    {
        CSVMIntBuffer buffer = CBufferFactory.allocSVMInt(context, triangleArraySize(), READ_ONLY);
        buffer.mapWriteBuffer(queue, intbuffer -> intbuffer.put(getTriangleFacesArray()));        
        return buffer;
    }
    
    public CSVMIntBuffer getCLSizeBufferSVM(CContext context, CCommandQueue queue)
    {
        CSVMIntBuffer buffer = CBufferFactory.allocSVMInt(context, 1, READ_ONLY);
        buffer.mapWriteBuffer(queue, intbuffer -> intbuffer.put(triangleSize()));        
        return buffer;
    }   
}
