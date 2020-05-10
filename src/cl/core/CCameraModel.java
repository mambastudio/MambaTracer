/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CRay;
import coordinate.model.CameraModel;

/**
 *
 * @author user
 */
public class CCameraModel extends CameraModel <CPoint3, CVector3, CRay>{    
    public CCameraModel(CPoint3 position, CPoint3 lookat, CVector3 up, float horizontalFOV) {
        super(position.copy(), lookat.copy(), up.copy(), horizontalFOV);
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
