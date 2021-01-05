/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.abstracts;

import cl.scene.CMesh;
import cl.struct.CBound;
import cl.struct.CIntersection;
import cl.struct.CNode;
import cl.struct.CRay;
import coordinate.generic.raytrace.AbstractAccelerator;
import wrapper.core.CMemory;

/**
 *
 * @author user
 */
public interface CAcceleratorInterface extends AbstractAccelerator<CRay, CIntersection, CMesh, CBound>
{

    public CMemory<CBound> getBounds();
    public CMemory<CNode> getNodes();
    default int getStartNodeIndex()
    {
        return 0;
    }
}
