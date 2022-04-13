package cl.abstracts;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import bitmap.core.AbstractDisplay;
import coordinate.generic.AbstractBound;
import coordinate.generic.AbstractMesh;
import coordinate.generic.raytrace.AbstractAccelerator;
import coordinate.model.CameraModel;

/**
 *
 * @author user
 * @param <A>
 * @param <D>
 * @param <M>
 * @param <MS>
 * @param <AC>
 * @param <BB>
 * @param <CM>
 * @param <CD>
 */
public interface RayDeviceInterface <
        A       extends MambaAPIInterface, 
        D       extends AbstractDisplay, 
        M       extends MaterialInterface, 
        MS      extends AbstractMesh,
        AC      extends AbstractAccelerator,
        BB      extends AbstractBound,
        CM      extends CameraModel,
        CD      extends CameraDataAbstract> {    
    public enum DeviceBuffer{
        IMAGE_BUFFER,
        GROUP_BUFFER,
        RENDER_BUFFER
    }
            
    public void setAPI(A api);
    
    public void set(MS mesh, AC bvhBuild);        
    public void setGlobalSize(int globalSize);
    public void setLocalSize(int localSize);
    
    default void start(){execute();}            
    public void execute();    
    public void pause();
    public void stop();
    public void resume();
    
    public boolean isPaused();
    public boolean isRunning();
    public boolean isStopped();
        
    default void updateImage(){}
    
    public void updateCamera();    
    public void setCamera(CD cameraData);    
    public CM getCameraModel(); 
    
    default BB getPriorityBound(){return null;}; //specific bound in mind, like selected by mouse
    default void setPriorityBound(BB bound){};
    default BB getBound(){return null;}; 
    
    //selected object or specific object bounds
    default BB getGroupBound(int value){return null;};
}
