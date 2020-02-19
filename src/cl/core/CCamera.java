/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CRay;
import coordinate.model.CameraModel;
import coordinate.struct.ByteStruct;

/**
 *
 * @author user
 */
public class CCamera extends CameraModel <CPoint3, CVector3, CRay>{    
    public CCamera(CPoint3 position, CPoint3 lookat, CVector3 up, float horizontalFOV) {
        super(position.copy(), lookat.copy(), up.copy(), horizontalFOV);
    }

    @Override
    public CCamera copy() {
        CCamera camera = new CCamera(position, lookat, up, fov);
        camera.setUp();
        return camera;
    }
    
    public CameraStruct getCameraStruct()
    {
        CameraStruct camera = new CameraStruct();
        
        camera.setPosition(position);
        camera.setLookat(lookat);
        camera.setUp(up);
        camera.setFov(fov);
        
        return camera;        
    }
    
    public CTransform getTransform()
    {
        CTransform transform = new CTransform();
        transform.setTransform(this.cameraTransform.m.m, this.cameraTransform.mInv.m);
        return transform;
    }
    
    public static class CameraStruct extends ByteStruct
    {
        public CPoint3 position; 
        public CPoint3 lookat;
        public CVector3 up;
        public CPoint2 dimension;
        public float fov;
        
        public CameraStruct()
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
    }  
    
    
    public boolean isSynched(CameraStruct cameraStruct)
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
