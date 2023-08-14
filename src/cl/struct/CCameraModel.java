/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.struct.CCamera;
import cl.struct.CRay;
import cl.data.CPoint3;
import cl.data.CVector3;
import coordinate.model.CameraModel;
import coordinate.transform.Transform;

/**
 *
 * @author user
 */
public class CCameraModel extends CameraModel <CPoint3, CVector3, CRay>{
    public CCameraModel(CPoint3 position, CPoint3 lookat, CVector3 up, float horizontalFOV) {
        super(position.copy(), lookat.copy(), up.copy(), horizontalFOV);
    }
    
    public void set(CPoint3 position, CPoint3 lookat, CVector3 up, float horizontalFOV)
    {
        this.position = position.copy();
        this.lookat = lookat.copy();
        this.up = up.copy();
        this.fov = horizontalFOV;
        this.cameraTransform = new Transform<>();
    }

    @Override
    public CCameraModel copy() {
        CCameraModel camera = new CCameraModel(position, lookat, up, fov);
        camera.setUp();
        return camera;
    }
    
    public CCamera getCameraStruct()
    {
        CCamera camera = new CCamera();        
        camera.setPosition(position);
        camera.setLookat(lookat);
        camera.setUp(up);
        camera.setFov(fov);
        
        return camera;        
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
