/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.shapes;

import cl.core.CNormalBVH;
import cl.core.CBoundingBox;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CRay;
import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CMaterial;
import coordinate.generic.AbstractMesh;
import coordinate.generic.raytrace.AbstractPrimitive;
import coordinate.list.CoordinateList;
import coordinate.list.IntList;
import coordinate.parser.attribute.MaterialT;
import wrapper.core.CBufferFactory;
import wrapper.core.CCommandQueue;
import wrapper.core.CContext;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
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
         CNormalBVH, 
         CBoundingBox> {
    
    CNormalBVH accelerator;
    CBoundingBox bounds;
    OpenCLPlatform configuration;
        
    public CMesh(OpenCLPlatform configuration)
    {
        this.configuration = configuration;
        
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
    public CNormalBVH getAccelerator() {
        return accelerator;
    }

    @Override
    public void buildAccelerator() {
        //this.accelerator = new CNormalBVH();
        //this.accelerator.build(this);
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
            
    public CFloatBuffer getCLPointsBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapFloat(name, context, queue, getPointArray(), READ_ONLY);          
    }
    
    public CFloatBuffer getCLNormalsBuffer(String name, CContext context, CCommandQueue queue)
    {        
        return CBufferFactory.wrapFloat(name, context, queue, getNormalArray(), READ_ONLY);          
    }
    
    public CIntBuffer getCLFacesBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapInt(name, context, queue, getTriangleFacesArray(), READ_ONLY);
    }
    
    public CIntBuffer getCLSizeBuffer(String name, CContext context, CCommandQueue queue)
    {
        return CBufferFactory.wrapInt(name, context, queue, new int[]{triangleSize()}, READ_ONLY);               
    }   
    
    public CStructBuffer<CMaterial> getCLMaterialBuffer(String name, CContext context, CCommandQueue queue)
    {        
        CStructBuffer<CMaterial> cmaterials = CBufferFactory.allocStruct(name, configuration.context(), CMaterial.class, this.getMaterialList().size(), READ_WRITE);       
        cmaterials.mapWriteBuffer(queue, materialArray -> {
            for(int i = 0; i<materialArray.length; i++)
            {
                MaterialT mat = this.getMaterialList().get(i);                 
                materialArray[i].setMaterial(mat);   
                
            }
        });
        return cmaterials;
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
