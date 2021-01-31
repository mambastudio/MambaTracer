/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CPoint2;
import cl.data.CPoint3;
import coordinate.generic.raytrace.AbstractIntersection;
import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CIntersection extends Structure implements AbstractIntersection{
        
    public CPoint3 p;
    public CPoint3 n;
    public CPoint3 d;    
    public CPoint2 uv;
    public int mat;
    public int id;
    public int hit;   
    
    public CIntersection()
    {        
        this.p = new CPoint3();
        this.n = new CPoint3();
        this.d = new CPoint3();        
        this.uv = new CPoint2();
        this.mat = 0;
        this.id = 0;
        this.hit = 0;
    }
    
    public void setMat(int mat)
    {
        this.mat = mat;
        this.refreshGlobalArray();
    }

    public void setHit(int hit)
    {
        this.hit = hit;
        this.refreshGlobalArray();
    }
    
    public int getHit()
    {
        this.refreshGlobalArray();
        return hit;
    } 
    
    public boolean isHit()
    {
        this.refreshGlobalArray();
        return hit == 1;
    }
}
