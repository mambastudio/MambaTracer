package cl.abstracts;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import bitmap.core.AbstractDisplay;
import bitmap.image.BitmapARGB;
import bitmap.image.BitmapRGBE;
import coordinate.utility.Value2Di;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Supplier;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import wrapper.core.OpenCLConfiguration;

/**
 *
 * @author user
 * @param <I>
 * @param <M>
 */
public interface MambaAPIInterface <
        M extends MaterialInterface, 
        I extends RenderControllerInterface>
{
    enum ImageType{
        RAYTRACE_IMAGE, 
        RENDER_IMAGE,
        OVERLAY_IMAGE,
        ALL_RAYTRACE_IMAGE
    };
    
    enum DeviceType{
        RAYTRACE,
        RENDER
    }
    
    enum AcceleratorType{
        PLOC,
        MEDIANCUT_BVH
    };
    
    public final StringProperty message = new SimpleStringProperty();
    
    //just the size of group size, but is depended on a local size power of 2
    public static int getNumOfGroups(int length, int LOCALSIZE)
    {
        int a = length/LOCALSIZE;
        int b = length%LOCALSIZE; //has remainder
        
        return (b > 0)? a + 1 : a;
            
    }
    
    //returns a global size of power of 2
    public static int getGlobal(int size, int LOCALSIZE)
    {
        if (size % LOCALSIZE == 0) { 
            return (int) ((Math.floor(size / LOCALSIZE)) * LOCALSIZE); 
        } else { 
            return (int) ((Math.floor(size / LOCALSIZE)) * LOCALSIZE) + LOCALSIZE; 
        } 
    }
    
    default void setMessage(String string)
    {
        Platform.runLater(()->message.set(string));
    }
    
    public void initOpenCLConfiguration();    
    public OpenCLConfiguration getConfigurationCL();
    
    public void init();
        
    default BitmapARGB getBitmap(ImageType imageType){return null;}; //TO DELETE
    default void setBitmap(ImageType name, BitmapARGB bitmap){}; //TO DELETE
    default void initBitmap(ImageType name){}; //TO DELETE    
    public <D extends AbstractDisplay> D getDisplay(Class<D> displayClass);
    public <D extends AbstractDisplay> void setDisplay(Class<D> displayClass, D display);    
    public Value2Di getImageSize(ImageType imageType);    
    public void setImageSize(ImageType name, int width, int height);
    
    default void readImageFromDevice(DeviceType device, ImageType imageType){}; //TO DELETE
    default void applyImage(ImageType name, Supplier<BitmapARGB> supply){}; //TO DELETE
    public default int getImageWidth(ImageType imageType){return getImageSize(imageType).x;}
    public default int getImageHeight(ImageType imageType){return getImageSize(imageType).y;}
    
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
    
    public void setDevicePriority(DeviceType device);
    public DeviceType getDevicePriority();
    public boolean isDevicePriority(DeviceType device);
        
    public RayDeviceInterface getDevice(DeviceType device);
    public void set(DeviceType device, RayDeviceInterface deviceImplementation);
    public I getController(String controller);
    public void set(String controller, I controllerImplementation);
    
    public void setMaterial(int index, M material);
    public M getMaterial(int index);
    
    public void setEnvironmentMap(BitmapRGBE bitmap);
    
    public default <T> T getObject(Supplier<T> supplier)
    {
        return supplier.get();
    }
    
    default LightManager getLightManager()
    {
        return null;
    }
}
