/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.CMaterialInterface;
import cl.core.CNormalBVH;
import cl.core.Overlay;
import cl.core.api.MambaAPIInterface;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.core.api.MambaAPIInterface.DeviceType.RENDER;
import static cl.core.api.MambaAPIInterface.ImageType.ALL_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.OVERLAY_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import cl.core.api.RayDeviceInterface;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.GROUP_BUFFER;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.IMAGE_BUFFER;
import cl.core.api.RenderControllerInterface;
import cl.core.device.RayDeviceMesh;
import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import cl.ui.mvc.viewmodel.MaterialEditorModel;
import coordinate.parser.OBJParser;
import coordinate.parser.attribute.MaterialT;
import coordinate.utility.Timer;
import coordinate.utility.Value2Di;
import filesystem.core.OutputFactory;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.function.Supplier;
import org.jocl.CL;
import wrapper.core.OpenCLPlatform;

/**
 *
 * @author user
 */
public final class TracerAPI implements MambaAPIInterface<IntBuffer, BlendDisplay, MaterialT> {
    
    //Opencl configuration for running single ray tracing program (might add one for future material editor)
    private OpenCLPlatform configuration = null;
    
    //The display to be used, is independent of ui since it can be file writer (jpg, png,...) display
    private BlendDisplay display = null;
     
    //Controller 
    private RenderControllerInterface controllerImplementation = null;
   
    //MaterialEditorModel
    public MaterialEditorModel materialEditorModel = new MaterialEditorModel();
    public CMaterialInterface cmat = null;
    
    //Overlay data for ui
    protected Overlay overlay;    
    protected BitmapARGB raytraceBitmap;
    protected BitmapARGB overlayBitmap;
    protected BitmapARGB renderBitmap;
    
    //Images dimension
    private Value2Di raytraceImageDimension;
    private Value2Di renderImageDimension;
    
    //ray casting device
    private RayDeviceInterface<TracerAPI, IntBuffer, BlendDisplay, MaterialT> deviceRaytrace;
    private RayDeviceInterface<TracerAPI, IntBuffer, BlendDisplay, MaterialT> deviceRender;    
    
    //mesh and accelerator
    private CMesh mesh;
    private CNormalBVH bvhBuild;
    
    public TracerAPI()
    {
         CL.setExceptionsEnabled(true);
         
        //opencl configuration
        this.initOpenCLConfiguration();
    }
    
    @Override
    public void init() {
       
        
        //dimension of images
        this.raytraceImageDimension = new Value2Di(800, 600);
        this.renderImageDimension = new Value2Di(800, 600);
        
        //create bitmap images
        this.initBitmap(ALL_IMAGE);
        
        //instantiate devices
        deviceRaytrace = new RayDeviceMesh();
        deviceRaytrace.setAPI(this);
        
        //default mesh
        this.initDefaultMesh();
                
        //start ray tracing
        deviceRaytrace.start();
    }

    @Override
    public final void initOpenCLConfiguration() {        
        if(configuration != null) 
            return;
        configuration = OpenCLPlatform.getDefault(CLSource.readFiles());        
        OutputFactory.print("name", configuration.device().getName());
        OutputFactory.print("type", configuration.device().getType());
        OutputFactory.print("vendor", configuration.device().getVendor());
        OutputFactory.print("speed", Long.toString(configuration.device().getSpeed()));
        
    }

    @Override
    public RayDeviceInterface getDevice(DeviceType device) {
        if(device.equals(RENDER))
            return deviceRender;
        else if(device.equals(RAYTRACE))
            return deviceRaytrace;
        return null;
    }

    @Override
    public OpenCLPlatform configurationCL() {
        return configuration;
    }

    @Override
    public Value2Di getImageSize(ImageType image) {
        switch (image) {
            case RENDER_IMAGE:
                return this.renderImageDimension;
            case RAYTRACE_IMAGE:
                return this.raytraceImageDimension;
            default:
                return null;
        }
    }

    @Override
    public void setImageSize(ImageType image, int width, int height) {
        switch (image) {
            case RENDER_IMAGE:
                this.renderImageDimension.set(width, height);                
            case RAYTRACE_IMAGE:
                this.raytraceImageDimension.set(width, height);
            default:
                break;
        }
        
    }

    @Override
    public BitmapARGB getBitmap(ImageType image) {
        switch (image) {
            case RAYTRACE_IMAGE:
                return this.raytraceBitmap;
            case RENDER_IMAGE:
                return this.renderBitmap;
            case OVERLAY_IMAGE:
                return this.overlayBitmap;
            default:
                return null;
        }
    }

    @Override
    public void initBitmap(ImageType image) {
        switch (image) {
            case RAYTRACE_IMAGE:
                raytraceBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y);
                overlayBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y, false);
                overlay = new Overlay(raytraceImageDimension.x, raytraceImageDimension.y);
                
                display.set(RAYTRACE_IMAGE.name(), raytraceBitmap); 
                break;
            case RENDER_IMAGE:
                renderBitmap = new BitmapARGB(renderImageDimension.x, renderImageDimension.y, true);
                break;
            case ALL_IMAGE:
                raytraceBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y, true);
                overlayBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y, false);
                overlay = new Overlay(raytraceImageDimension.x, raytraceImageDimension.y);
                renderBitmap = new BitmapARGB(renderImageDimension.x, renderImageDimension.y, true);
                
                display.set(RAYTRACE_IMAGE.name(), raytraceBitmap); 
                break;
            default:
                break;
        }
    }

    @Override
    public void setBlendDisplay(BlendDisplay display) {
        this.display = display;
    }
    
    @Override
    public BlendDisplay getBlendDisplay()
    {
        return display;
    }

    @Override
    public void render(DeviceType device) {
        if(device.equals(RAYTRACE))
            deviceRaytrace.start();
        else if(device.equals(RENDER))
            deviceRender.start();
    }

    @Override
    public void readImageFromDevice(DeviceType device, ImageType image) {        
        if(device.equals(RAYTRACE))         
            if(image == RAYTRACE_IMAGE)
            {                
                deviceRaytrace.readBuffer(IMAGE_BUFFER, buffer-> {            
                    this.getBitmap(image).writeColor(buffer.array(), 0, 0, raytraceImageDimension.x, raytraceImageDimension.y);
                    this.display.imageFill(RAYTRACE_IMAGE.name(), this.getBitmap(RAYTRACE_IMAGE));                    
                });
            }
            else if(image == OVERLAY_IMAGE)
                deviceRaytrace.readBuffer(GROUP_BUFFER, buffer-> {
                    overlay.copyToArray(buffer.array()); 
                });
            else if(image == ALL_IMAGE)
            {
                deviceRaytrace.readBuffer(IMAGE_BUFFER, buffer-> {            
                    this.getBitmap(RAYTRACE_IMAGE).writeColor(buffer.array(), 0, 0, raytraceImageDimension.x, raytraceImageDimension.y);
                    this.display.imageFill(RAYTRACE_IMAGE.name(), this.getBitmap(RAYTRACE_IMAGE));
                });
                deviceRaytrace.readBuffer(GROUP_BUFFER, buffer-> {
                    overlay.copyToArray(buffer.array()); 
                });
            }
    }

    @Override
    public void initMesh(Path path) 
    {
        //load mesh and init mesh variables
        mesh = new CMesh(configuration);
        OBJParser parser = new OBJParser();    
        Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh)); //Time parsing
        OutputFactory.print("scene parse time", parseTime.toString());
        mesh.initCLBuffers();
        
        //display material in ui
        controllerImplementation.displaySceneMaterial(mesh.getMaterialList());
        
        //build accelerator
        Timer buildTime = Timer.timeThis(() -> {                                   //Time building
            this.bvhBuild = new CNormalBVH(configuration);
            this.bvhBuild.build(mesh);      
        });
        OutputFactory.print("bvh build time", buildTime.toString());
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
    }
    
    @Override
    public void initDefaultMesh()
    {
        //A simple cube
        String cube =   "o Cube\n" +
                        "v 1.000000 -1.000000 -1.000000\n" +
                        "v 1.000000 -1.000000 1.000000\n" +
                        "v -1.000000 -1.000000 1.000000\n" +
                        "v -1.000000 -1.000000 -1.000000\n" +
                        "v 1.000000 1.000000 -0.999999\n" +
                        "v 0.999999 1.000000 1.000001\n" +
                        "v -1.000000 1.000000 1.000000\n" +
                        "v -1.000000 1.000000 -1.000000\n" +
                        "vn 0.000000 -1.000000 0.000000\n" +
                        "vn 0.000000 1.000000 0.000000\n" +
                        "vn 1.000000 0.000000 0.000000\n" +
                        "vn -0.000000 0.000000 1.000000\n" +
                        "vn -1.000000 -0.000000 -0.000000\n" +
                        "vn 0.000000 0.000000 -1.000000\n" +
                        "s off\n" +
                        "f 2//1 3//1 4//1\n" +
                        "f 8//2 7//2 6//2\n" +
                        "f 5//3 6//3 2//3\n" +
                        "f 6//4 7//4 3//4\n" +
                        "f 3//5 7//5 8//5\n" +
                        "f 1//6 4//6 8//6\n" +
                        "f 1//1 2//1 4//1\n" +
                        "f 5//2 8//2 6//2\n" +
                        "f 1//3 5//3 2//3\n" +
                        "f 2//4 6//4 3//4\n" +
                        "f 4//5 3//5 8//5\n" +
                        "f 5//6 1//6 8//6";
        
        //load mesh and init mesh variables
        mesh = new CMesh(configurationCL());   
        OBJParser parser = new OBJParser();        
        parser.readString(cube, mesh);
        mesh.initCLBuffers();
        
        //display material in ui
        controllerImplementation.displaySceneMaterial(mesh.getMaterialList());
        
        //build accelerator
        this.bvhBuild = new CNormalBVH(configurationCL());
        this.bvhBuild.build(mesh);    
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
    }

    @Override
    public void startDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.start();
        else if(device.equals(RENDER))
            this.deviceRender.start();
    }

    @Override
    public void pauseDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.pause();
        else if(device.equals(RENDER))
            this.deviceRender.pause();
    }

    @Override
    public void stopDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.stop();
        else if(device.equals(RENDER))
            this.deviceRender.stop();
    }

    @Override
    public void set(DeviceType device, RayDeviceInterface deviceImplementation) {
        if(device.equals(RAYTRACE))
        {
            this.deviceRaytrace = deviceImplementation;
            this.deviceRaytrace.setAPI(this);
        }
        else if(device.equals(RENDER))
        {
            this.deviceRender = deviceImplementation;
            this.deviceRender.setAPI(this);
        }
    }

    @Override
    public RenderControllerInterface getController(String controller) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void set(String controller, RenderControllerInterface controllerImplementation) {
        this.controllerImplementation = controllerImplementation;
        this.controllerImplementation.setAPI(this);
        
    }

    @Override
    public void applyImage(ImageType name, Supplier<BitmapARGB> supply) {
        setBitmap(name, supply.get());
    }

    @Override
    public void setBitmap(ImageType name, BitmapARGB bitmap) {
        switch (name) {
            case RAYTRACE_IMAGE:
                this.raytraceBitmap = bitmap;
            case RENDER_IMAGE:
                this.renderBitmap = bitmap;
            case ALL_IMAGE:
                this.overlayBitmap = bitmap;
            default:
                break;
        }
    }

    @Override
    public void resumeDevice(DeviceType device) {
        switch (device) {
            case RAYTRACE:
                this.deviceRaytrace.resume();
                break;
            case RENDER:
                this.deviceRender.resume();
                break;
            default:
                break;
        }
    }

    @Override
    public boolean isDeviceRunning(DeviceType device) {
        switch (device) {
            case RAYTRACE:
                return this.deviceRaytrace.isRunning();
            case RENDER:
                return false;
            default:
                throw new UnsupportedOperationException(device+ " not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    @Override
    public int getGlobalSizeForDevice(DeviceType device) {
        switch (device) {
            case RAYTRACE:
                return this.raytraceImageDimension.x * this.raytraceImageDimension.y;
            case RENDER:
                return this.renderImageDimension.x * this.renderImageDimension.y;
            default:
                return -1;
        }
    }

    
}
