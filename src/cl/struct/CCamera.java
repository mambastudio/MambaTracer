/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.abstracts.CameraDataAbstract;
import cl.data.CPoint2;
import cl.data.CPoint3;
import cl.data.CVector3;

/**
 *
 * @author user
 */
public class CCamera extends CameraDataAbstract {
    public CPoint3 position; 
    public CPoint3 lookat;
    public CVector3 up;
    public CPoint2 dimension;
    public float fov;

    public CCamera()
    {
        position = new CPoint3();
        lookat = new CPoint3();
        dimension = new CPoint2();
        up = new CVector3();            
    }

    public void setPosition(CPoint3 position)
    {
        this.position = position;
        this.refreshGlobalArray();
    }

    public void setLookat(CPoint3 lookat)
    {
        this.lookat = lookat;
        this.refreshGlobalArray();
    }

    public void setUp(CVector3 up)
    {
        this.up = up;
        this.refreshGlobalArray();
    }

    public void setDimension(CPoint2 dimension)
    {
        this.dimension = dimension;
        this.refreshGlobalArray();
    }

    public void setFov(float fov)
    {
        this.fov = fov;
        this.refreshGlobalArray();
    }
    
    public boolean isSynched(CCamera cameraStruct)
    {
        float x  = position.x;
        float x1 = cameraStruct.position.get(0);
        float y  = position.y;
        float y1 = cameraStruct.position.get(1);
        float z  = position.z;
        float z1 = cameraStruct.position.get(2);
        
        return (x == x1) && (y == y1) &&  (z == z1);
    }
}
