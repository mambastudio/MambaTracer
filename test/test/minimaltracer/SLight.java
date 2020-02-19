/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class SLight extends ByteStruct{
    public int faceId;
    
    public CPoint3 p;
    public CVector3 d;
    
    public SLight()
    {
        faceId = 0;
        p = new CPoint3();
        d = new CVector3();
        
    }
    
    public SLight(int faceId)
    {
        this.faceId = faceId;
        
        p = new CPoint3();
        d = new CVector3();
    }
    
    public int getFaceId()
    {
        this.refreshGlobalArray();
        return faceId;
    } 
}
