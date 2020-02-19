/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
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
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class SMesh extends AbstractMesh<CPoint3, CVector3, CPoint2> implements AbstractPrimitive
        <CPoint3,         
         SRay, 
         SIsect, 
         SNormalBVH, 
         SBound> {
       
    private final OpenCLPlatform configuration;
    
    //opencl mesh bound
    private final SBound bounds;
    
    //opencl mesh data
    private CFloatBuffer pointsBuffer = null;
    private CFloatBuffer normalsBuffer = null;
    private CIntBuffer facesBuffer = null;
    private CIntBuffer sizeBuffer = null;
    
    //materials
    private CStructTypeBuffer<SMaterial> cmaterials = null;
        
    public SMesh(OpenCLPlatform configuration)
    {
        this.configuration = configuration;
        
        points = new CoordinateList(CPoint3.class);
        normals = new CoordinateList(CVector3.class);
        texcoords = new CoordinateList(CPoint2.class);
        triangleFaces = new IntList();
        bounds = new SBound();
    }

    @Override
    public int getCount() {
        return triangleSize();
    }

    @Override
    public SBound getBound(int primID) {
        SBound bbox = new SBound();
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
    public SBound getBound() {
        return bounds;
    }

    @Override
    public boolean intersect(SRay r, int primID, SIsect isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(SRay r, int primID) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersect(SRay r, SIsect isect) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean intersectP(SRay r) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public SNormalBVH getAccelerator() {
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
        cmaterials = CBufferFactory.allocStructType("materials", configuration.context(), SMaterial.class, this.getMaterialList().size(), READ_WRITE);       
        cmaterials.mapWriteBuffer(configuration.queue(), materialArray -> {
            for(int i = 0; i<materialArray.size(); i++)
            {
                MaterialT mat = this.getMaterialList().get(i);   
                materialArray.get(i).setMaterial(mat);               
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
        
    public CStructTypeBuffer<SMaterial> clMaterials()
    {
        return cmaterials;
    }
        
    public void setMaterial(int index, SMaterial material)    
    {
        this.cmaterials.mapWriteBuffer(configuration.queue(), materialArray -> {   
            materialArray.set(material, index);
        });        
    }
    
    public SMaterial getMaterial(int index)
    {
        return cmaterials.get(index);
    }
   
}
