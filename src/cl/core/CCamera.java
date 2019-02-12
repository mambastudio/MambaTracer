/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import cl.core.data.struct.CRay;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.model.CameraModel;
import org.jocl.struct.CLTypes.cl_float4;
import org.jocl.struct.Struct;

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
        camera.position = position.getFloatCL4();
        camera.lookat = lookat.getFloatCL4();
        camera.up = up.getFloatCL4();
        camera.fov = fov;
        return camera;        
    }
    
    public static class CameraStruct extends Struct
    {
        public cl_float4 position;
        public cl_float4 lookat;
        public cl_float4 up;
        public float fov;
    }
    
}
