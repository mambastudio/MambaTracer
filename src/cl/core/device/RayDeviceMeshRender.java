/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CNormalBVH;
import cl.core.api.MambaAPIInterface;
import static cl.core.api.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.core.api.RayDeviceInterface;
import cl.shapes.CMesh;
import coordinate.parser.attribute.MaterialT;
import wrapper.core.CallBackFunction;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class RayDeviceMeshRender implements RayDeviceInterface {
    //API
    MambaAPIInterface api;
    
    //frame buffer
    private CIntBuffer frameBuffer = null;
    private CFloatBuffer frameCountBuffer = null;
    private CFloatBuffer accumBuffer = null;
    
    //count    
    private CIntBuffer   countBuffer = null;    
    
    //global and local size
    private int globalSize, localSize;
    
    public RayDeviceMeshRender()
    {
        
    }    
        
    public void initBuffers()
    {
        //init frame buffer
        frameBuffer.mapWriteBuffer(api.configurationCL().queue(), buffer -> {
           for(int i = 0; i<buffer.capacity(); i++)
               buffer.put(0);
        });        
        
        //init accumulation buffer
        accumBuffer.mapWriteBuffer(api.configurationCL().queue(), buffer -> {           
           for(int i = 0; i<buffer.capacity(); i++)
               buffer.put(0);
        });
        
        //init frame count
        frameCountBuffer.mapWriteValue(api.configurationCL().queue(), 0);
      
        //init render bitmap
        api.initBitmap(RENDER_IMAGE);        
    }   
    
    @Override
    public void start()
    {/*
        if(renderThread.isPaused())
            renderThread.resumeExecution();
        else if(renderThread.isTerminated()) 
        {
            initBuffers();
            renderThread.restartExecution();
        }        
        else
        {
            initBuffers();            
            renderThread.startExecution(() -> {
                
                //set ray count and hit count to zero and generate camera rays
                rayCount.mapWriteValue(configuration.queue(), 0);     
                hitCount.mapWriteValue(configuration.queue(), 0);
                configuration.queue().put1DRangeKernel(initCameraRaysKernel, globalSize, localSize); 
                
                //start tracing path
                for(int i = 0; i<1; i++)
                {
                    //intersect scene
                    configuration.queue().put1DRangeKernel(intersectPrimitivesKernel, globalSize, localSize); 
                    renderThread.chill();
                    
                    //if hit count is greater than zero
                    if(hitCount.mapReadValue(configuration.queue()) > 0)
                    {
                        //in case you hit light mesh
                        configuration.queue().put1DRangeKernel(lightHitKernel, globalSize, localSize);   
                        
                        //sample brdf
                        configuration.queue().put1DRangeKernel(sampleBRDFKernel, globalSize, localSize);
                        
                        //set ray
                    }
                }
                
                //increment frame count by 1
                frameCountBuffer.mapWriteValue(configuration.queue(), frameCountBuffer.mapReadValue(configuration.queue()) + 1);             

                //update frame
                configuration.queue().put1DRangeKernel(updateFrameImageKernel, globalSize, localSize);
                
                //write pixels to display
                frameBuffer.mapReadBuffer(configuration.queue(), buffer -> {
                    int fwidth = 800; int fheight = 700;
                    RenderViewModel.renderBitmap.writeColor(buffer.array(), 0, 0, fwidth, fheight);                    
                    RenderViewModel.display.imageFill("render", RenderViewModel.renderBitmap);
                });  
                                
                renderThread.chill();
            });
        }
        */
    }

    @Override
    public void setAPI(MambaAPIInterface api) {
        this.api = api;
    }

    @Override
    public void set(CMesh mesh, CNormalBVH bvhBuild) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setMaterial(int index, MaterialT aterial) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void execute() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void updateCamera() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CCamera getCamera() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getTotalSize() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBoundingBox getBound() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setLocalSize(int local) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void pause() {
        
    }

    @Override
    public void stop() {
        
    }

    @Override
    public boolean isPaused() {
        return true;
    }

    @Override
    public boolean isRunning() {
        return false;
    }

    @Override
    public void setCamera(CCamera camera) {
        
    }

    @Override
    public void resume() {
       
    }

    @Override
    public void setGlobalSize(int globalSize) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void readBuffer(DeviceBuffer name, CallBackFunction callback) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBoundingBox getGroupBound(int value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean isStopped() {
        return true;
    }

    @Override
    public void setShadeType(ShadeType type) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public ShadeType getShadeType() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
