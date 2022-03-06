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
public class CPath extends Structure{
    public CPoint3  throughput;
    public CPoint3  hitpoint;
    public int      pathlength;
    public boolean  lastSpecular;
    public float    lastPdfW;
    public boolean  active;
    public CBsdf    bsdf;
    
    public CPath()
    {
        throughput   = new CPoint3();
        hitpoint     = new CPoint3();
        pathlength   = 0;
        lastSpecular = true;
        lastPdfW     = 1;
        active       = false;
        bsdf         = new CBsdf();
    }
}
