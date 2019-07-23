/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.api;

import bitmap.display.BlendDisplay;
import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CNormalBVH;
import cl.shapes.CMesh;
import coordinate.parser.attribute.MaterialT;
import java.nio.Buffer;
import wrapper.core.CallBackFunction;

/**
 *
 * @author user
 * @param <A>
 * @param <B>
 * @param <D>
 * @param <M>
 */
public interface RayDeviceInterface <A extends MambaAPIInterface, B extends Buffer, D extends BlendDisplay, M extends MaterialT> {    
    public enum DeviceBuffer{
        IMAGE_BUFFER,
        GROUP_BUFFER
    }
    
    public void setAPI(A api);
    
    public void set(CMesh mesh, CNormalBVH bvhBuild);        
    public void setGlobalSize(int globalSize);
    public void setLocalSize(int localSize);
    
    default void start(){execute();}            
    public void execute();    
    public void pause();
    public void stop();
    public void resume();
    public boolean isPaused();
    public boolean isRunning();
    
    public void readBuffer(DeviceBuffer name, CallBackFunction<B> callback);   
    
    public void setMaterial(int index, M material);
    public void updateCamera();    
    public void setCamera(CCamera camera);    
    public CCamera getCamera(); 
    public int getTotalSize();    
    public CBoundingBox getBound(); 
    public CBoundingBox getGroupBound(int value);
    
   
}
