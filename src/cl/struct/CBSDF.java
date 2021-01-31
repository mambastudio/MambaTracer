/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CPoint3;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CBSDF extends Structure {
    public int materialID;              //material id
    
    public boolean isPortal;
    
    public int paramLevel;
    public CSurfaceParameter param;
    
    public CFrame frame;                //local frame of reference
    public CPoint3 localDirFix;       //incoming (fixed) incoming direction, in local
    
    public CBSDF()
    {
        materialID = 0;
        isPortal = false;
        paramLevel = 0;
        param = new CSurfaceParameter();
        frame = new CFrame();
        localDirFix = new CPoint3();
    }
}
