/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;


import cl.core.data.struct.Bound;
import cl.core.data.struct.Flag;
import cl.core.data.struct.Morton;
import cl.core.data.struct.Node;
import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import cl.shapes.CMesh;
import coordinate.generic.raytrace.AbstractAccelerator;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class CKarrasBVH implements AbstractAccelerator<CRay, 
                                                        CIntersection, 
                                                        CMesh, 
                                                        CBoundingBox> 
{
    CMesh                       mesh;
    CStructIntArray<Node>       cnodes;
    CStructFloatArray<Bound>    cbounds;
   
    OpenCLPlatform configuration;
    
    public CKarrasBVH(OpenCLPlatform configuration)
    {
        this.configuration = configuration;        
    }
    
    @Override
    public void build(CMesh primitives)
    {              
        this.mesh = primitives;
        
        int leafS = primitives.getCount();              // n
        int nodeS = leafS - 1;                          // n - 1 
        int tSize = nodeS + leafS;                      //2n - 1        
             
        //preparation data
        CStructIntArray<Morton> cmortons    = new CStructIntArray(configuration, Morton.class, leafS, "mortons", READ_WRITE);
        CStructIntArray<Flag>   cflags      = new CStructIntArray(configuration, Flag.class, nodeS, "flags", READ_WRITE);
        
        //init mesh
        CFloatBuffer cpoints = mesh.getCLPointsBuffer("points", configuration.context(), configuration.queue());
        CIntBuffer cfaces = mesh.getCLFacesBuffer("faces", configuration.context(), configuration.queue());
        CIntBuffer csize = mesh.getCLSizeBuffer("size", configuration.context(), configuration.queue());  
                
        //data for spatial index accelerator
        cnodes  = new CStructIntArray(configuration, Node.class, tSize, "nodes", READ_WRITE);
        cbounds = new CStructFloatArray(configuration, Bound.class, tSize, "bounds", READ_WRITE);
        
        //calculate mortons
        CKernel calculateMortonKernel = configuration.program().createKernel("calculateMorton", cpoints, cfaces, csize, cmortons.getCBuffer()); 
        configuration.queue().put1DRangeKernel(calculateMortonKernel, leafS, 1);
                
        //sort mortons   
        cmortons.transferFromDeviceToBuffer();
        Morton.sort(cmortons.getStructArray());        
        cmortons.transferFromBufferToDevice(); 
        
        //emit bvh tree
        CKernel emitHierarchyKernel = configuration.program().createKernel("emitHierarchy", cpoints, cfaces, csize, cmortons.getCBuffer(), cnodes.getCBuffer(), cbounds.getCBuffer()); 
        configuration.queue().put1DRangeKernel(emitHierarchyKernel, leafS, 1);
        
        //refit bounds for nodes
        CKernel refitBoundsKernel = configuration.program().createKernel("refitBounds", csize, cflags.getCBuffer(), cnodes.getCBuffer(), cbounds.getCBuffer()); 
        configuration.queue().put1DRangeKernel(refitBoundsKernel, leafS, 1);     
        
        cbounds.transferFromDeviceToBuffer();
        cnodes.transferFromDeviceToBuffer();
        
        //for(int i = 0; i<cbounds.getSize(); i++)
        //    System.out.println(cbounds.get(i));
    }
    
    public CIntBuffer getNodes()
    {
        return cnodes.getCBuffer();
    }
    
    public CFloatBuffer getBounds()
    {
        return cbounds.getCBuffer();
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
