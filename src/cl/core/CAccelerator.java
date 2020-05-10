/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.struct.CBound;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CNode;
import cl.core.data.struct.CRay;
import cl.core.data.struct.array.CStructFloatArray;
import cl.core.data.struct.array.CStructIntArray;
import cl.shapes.CMesh;
import coordinate.generic.raytrace.AbstractAccelerator;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public interface CAccelerator extends AbstractAccelerator<CRay, CIntersection, CMesh, CBoundingBox> 
{
    public CFloatBuffer getCBounds();    
    public CStructFloatArray<CBound> getBounds();    
    public CIntBuffer getCNodes();    
    public CStructIntArray<CNode> getNodes();
    default int getStartNodeIndex()
    {
        return 0;
    }
}
