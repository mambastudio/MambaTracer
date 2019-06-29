/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import cl.shapes.CMesh;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class RayDeviceMeshRenderBind {
    private OpenCLPlatform configuration;
    
    //frame buffer
    private CIntBuffer frameBuffer = null;
    private CFloatBuffer frameCountBuffer = null;
    private CFloatBuffer accumBuffer = null;
    
    //count    
    private CIntBuffer   countBuffer = null;    
    
    //global and local size
    private int globalSize, localSize;
    
    public RayDeviceMeshRenderBind()
    {
        
    }
    
    public void bindPlatform(OpenCLPlatform configuration, int width, int height)
    {
        this.configuration = configuration;
    }
    
    public void render(CMesh mesh)
    {
        
    }
}
