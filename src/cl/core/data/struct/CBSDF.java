/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.data.CPoint3;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class CBSDF extends ByteStruct {
    public int materialID;              //material id
    public CFrame frame;                //local frame of reference
    public CPoint3 localDirFix;       //incoming (fixed) incoming direction, in local
    
    public CBSDF()
    {
        materialID = 0;
        frame = new CFrame();
        localDirFix = new CPoint3();
    }
}
