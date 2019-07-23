/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.api;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import coordinate.parser.attribute.MaterialT;
import coordinate.utility.Value2Di;
import java.net.URI;
import java.nio.Buffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Supplier;
import wrapper.core.OpenCLPlatform;

/**
 *
 * @author user
 * @param <B>
 * @param <D>
 * @param <M>
 */
public interface MambaAPIInterface <B extends Buffer, D extends BlendDisplay, M extends MaterialT>
{
    enum ImageType{
        RAYTRACE_IMAGE, 
        RENDER_IMAGE,
        OVERLAY_IMAGE,
        ALL_IMAGE
    };
    
    enum DeviceType{
        RAYTRACE,
        RENDER
    }
    
    enum RenderType{};
    
    public void initOpenCLConfiguration();    
    public OpenCLPlatform configurationCL();
    
    public void init();
        
    public BitmapARGB getBitmap(ImageType name);
    public void setBitmap(ImageType name, BitmapARGB bitmap);
    public void initBitmap(ImageType name);
    public D getBlendDisplay();
    public void setBlendDisplay(D display);    
    public Value2Di getImageSize(ImageType name);    
    public void setImageSize(ImageType name, int width, int height);
    public void readImageFromDevice(DeviceType device, ImageType image);
    public void applyImage(ImageType name, Supplier<BitmapARGB> supply);
    public default int getImageWidth(ImageType name){return getImageSize(name).x;}
    public default int getImageHeight(ImageType name){return getImageSize(name).y;}
    
    public int getGlobalSizeForDevice(DeviceType device);
    
    public void render(DeviceType device);
    
    public default void initMesh(String uri) {initMesh(Paths.get(uri));}    
    public default void initMesh(URI uri)    {initMesh(Paths.get(uri));}       
    public default void initDefaultMesh(){}
    public void initMesh(Path path);
    
    public void startDevice(DeviceType device);
    public void pauseDevice(DeviceType device);
    public void stopDevice(DeviceType device);
    public void resumeDevice(DeviceType device);
    public boolean isDeviceRunning(DeviceType device);
        
    public RayDeviceInterface getDevice(DeviceType device);
    public void set(DeviceType device, RayDeviceInterface deviceImplementation);
    public RenderControllerInterface getController(String controller);
    public void set(String controller, RenderControllerInterface controllerImplementation);
    
    public default <T> T getObject(Supplier<T> supplier)
    {
        return supplier.get();
    }
}
