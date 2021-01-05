/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.scene;

import cl.struct.CMaterial;
import cl.struct.CFace;
import cl.struct.CIntersection;
import cl.struct.CRay;
import cl.struct.CBound;
import cl.data.CPoint2;
import cl.data.CPoint3;
import cl.data.CVector3;
import coordinate.generic.AbstractMesh;
import coordinate.generic.raytrace.AbstractPrimitive;
import coordinate.list.CoordinateFloatList;
import coordinate.list.IntList;
import coordinate.parser.attribute.MaterialT;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CMesh extends AbstractMesh<CPoint3, CVector3, CPoint2> implements AbstractPrimitive
        <CPoint3,         
         CRay, 
         CIntersection, 
         CNormalBVH, 
         CBound> {
       
    private final OpenCLConfiguration configuration;
    
    //opencl mesh bound
    private final CBound bounds;
    
    //opencl mesh data
    private CMemory<CPoint3> pointsBuffer = null;
    private CMemory<CVector3> normalsBuffer = null;
    private CMemory<CFace> facesBuffer = null;
    private CMemory<IntValue> sizeBuffer = null;
    
    //materials
    private CMemory<CMaterial> cmaterialsc = null;
        
    public CMesh(OpenCLConfiguration configuration)
    {
        this.configuration = configuration;
        
        points = new CoordinateFloatList(CPoint3.class);
        normals = new CoordinateFloatList(CVector3.class);
        texcoords = new CoordinateFloatList(CPoint2.class);
        triangleFaces = new IntList();
        bounds = new CBound();
    }

    @Override
    public int getCount() {
        return triangleSize();
    }

    @Override
    public CBound getBound(int primID) {
        CBound bbox = new CBound();
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
    public CBound getBound() {
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
        pointsBuffer = configuration.createFromF(CPoint3.class, getPointArray(), READ_ONLY); 
        normalsBuffer = configuration.createFromF(CVector3.class, getNormalArray(), READ_ONLY);
        facesBuffer = configuration.createFromI(CFace.class, getTriangleFacesArray(), READ_ONLY);
        sizeBuffer = configuration.createFromI(IntValue.class, new int[]{triangleSize()}, READ_ONLY);        
        cmaterialsc = configuration.createBufferB(CMaterial.class, this.getMaterialList().size(), READ_WRITE); 
        cmaterialsc.mapWriteIterator(materialArray -> {
            int i = 0;
            for(CMaterial cmat : materialArray)
            {
                cmat.setMaterial(new CMaterial()); //init array like any other opencl array
                MaterialT mat = this.getMaterialList().get(i);   
                cmat.setMaterial(mat);                
                i++;
            }
        });        
    }
        
    public CMemory<CPoint3> clPoints()
    {
        return pointsBuffer;
    }
    
    public CMemory<CVector3> clNormals()
    {
        return normalsBuffer;
    }
    
    public CMemory<CFace> clFaces()
    {
        return facesBuffer;
    }
    
    public CMemory<IntValue> clSize()
    {
        return sizeBuffer;
    }
        
    public CMemory<CMaterial> clMaterials()
    {
        cmaterialsc.transferFromDevice();
        return cmaterialsc;
    }
        
    public void setMaterial(int index, CMaterial material)    
    {
        cmaterialsc.set(index, material);
        cmaterialsc.transferToDevice();
    }
    
    public CMaterial getMaterial(int index)
    {
        return cmaterialsc.get(index);
    }
   
}
