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
public class CBsdf extends Structure{
    public  CSurfaceParameter2  param;     //chosen surface

    public  CFrame frame;                 //local frame of reference

    public  CPoint3 localDirFix;          //incoming (fixed) incoming direction, in local
    public  CPoint3 localGeomN;           //geometry normal (without normal shading)
   
    public  int materialID;              //material id (Check if necessary, if not remove)
   
    public  CComponentProbabilities probabilities; //!< Sampling probabilities
    
    public CBsdf()
    {
        param           = new CSurfaceParameter2();
        frame           = new CFrame();
        localDirFix     = new CPoint3();
        localGeomN      = new CPoint3();
        materialID      = 0;
        probabilities   = new CComponentProbabilities();
    }
}
