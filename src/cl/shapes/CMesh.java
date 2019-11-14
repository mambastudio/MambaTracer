/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.shapes;

import cl.core.CNormalBVH;
import cl.core.CBoundingBox;
import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CMaterial;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CRay;
import coordinate.generic.AbstractMesh;
import coordinate.generic.raytrace.AbstractPrimitive;
import coordinate.list.CoordinateList;
import coordinate.list.IntList;
import coordinate.parser.attribute.MaterialT;
import wrapper.core.CBufferFactory;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;

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
       
    private final OpenCLPlatform configuration;
    
    //opencl mesh bound
    private final CBoundingBox bounds;
    
    //opencl mesh data
    private CFloatBuffer pointsBuffer = null;
    private CFloatBuffer normalsBuffer = null;
    private CIntBuffer facesBuffer = null;
    private CIntBuffer sizeBuffer = null;
    
    //materials
    private CStructBuffer<CMaterial> cmaterials = null;
    
        
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
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
    
    public void initCLBuffers()
    {
        CResourceFactory.releaseMemory("points", "normals", "faces", "size", "materials");
        pointsBuffer = CBufferFactory.wrapFloat("points", configuration.context(), configuration.queue(), getPointArray(), READ_ONLY);
        normalsBuffer = CBufferFactory.wrapFloat("normals", configuration.context(), configuration.queue(), getNormalArray(), READ_ONLY); 
        facesBuffer = CBufferFactory.wrapInt("faces", configuration.context(), configuration.queue(), getTriangleFacesArray(), READ_ONLY);
        sizeBuffer = CBufferFactory.wrapInt("size", configuration.context(), configuration.queue(), new int[]{triangleSize()}, READ_ONLY);
        cmaterials = CBufferFactory.allocStruct("materials", configuration.context(), CMaterial.class, this.getMaterialList().size(), READ_WRITE);       
        cmaterials.mapWriteBuffer(configuration.queue(), materialArray -> {
            for(int i = 0; i<materialArray.length; i++)
            {
                MaterialT mat = this.getMaterialList().get(i);                 
                materialArray[i].setMaterial(mat);   
                
            }
        });
    }
    
    public CFloatBuffer clPoints()
    {
        return pointsBuffer;
    }
    
    public CFloatBuffer clNormals()
    {
        return normalsBuffer;
    }
    
    public CIntBuffer clFaces()
    {
        return facesBuffer;
    }
    
    public CIntBuffer clSize()
    {
        return sizeBuffer;
    }
    
    public CStructBuffer<CMaterial> clMaterials()
    {
        return cmaterials;
    }
        
    public void setMaterial(int index, CMaterial material)    
    {
        this.cmaterials.mapWriteBuffer(configuration.queue(), materialArray -> {            
            materialArray[index] = material;            
        });
    }
   
}
