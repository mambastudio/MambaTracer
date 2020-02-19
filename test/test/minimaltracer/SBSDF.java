/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.CPoint3;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class SBSDF extends ByteStruct {
    public int materialID;              //material id
    public SFrame frame;                //local frame of reference
    public CPoint3 localDirFix;       //incoming (fixed) incoming direction, in local
    
    public SBSDF()
    {
        materialID = 0;
        frame = new SFrame();
        localDirFix = new CPoint3();
    }
}
