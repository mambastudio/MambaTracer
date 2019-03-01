/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.api;

import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.data.struct.CMaterial;
import java.net.URI;
import java.nio.IntBuffer;
import java.nio.file.Path;
import wrapper.core.CallBackFunction;

/**
 *
 * @author user
 */
public interface RayDeviceInterface {    
    /**
     * Initialize the javadoc with the appropriate width and height image
     * with a global and local size memory
     *     
     * @param width width of the image
     * @param height height of the image
     * @param globalSize global size of the memory device
     * @param localSize local size of the memory device
     */
    public void init(int width, int height, int globalSize, int localSize);
    public void initMesh(String uri);
    public void initMesh(URI uri);        
    public void initMesh(Path path);
       
    public void execute();
    
    public void readImageBuffer(String name, CallBackFunction<IntBuffer> callback);   
    public void setMaterial(int index, CMaterial material);
    public CMaterial getMaterial(int index);
    public void updateCamera();    
    public CCamera getCamera(); 
    public int getTotalSize();    
    public CBoundingBox getBound(); 
    
   
}
