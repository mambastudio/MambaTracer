/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.CAccelerator;
import cl.core.CMaterialInterface;
import cl.core.accelerator.CNormalBVH;
import cl.core.accelerator.CPlocBVH;
import cl.core.Overlay;
import cl.core.api.MambaAPIInterface;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.core.api.MambaAPIInterface.DeviceType.RENDER;
import static cl.core.api.MambaAPIInterface.ImageType.OVERLAY_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.core.api.RayDeviceInterface;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.GROUP_BUFFER;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.IMAGE_BUFFER;
import cl.core.api.RenderControllerInterface;
import cl.core.device.RayDeviceMesh;
import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import cl.ui.mvc.viewmodel.MaterialEditorModel;
import coordinate.parser.obj.OBJParser;
import coordinate.parser.attribute.MaterialT;
import coordinate.utility.Timer;
import coordinate.utility.Value2Di;
import filesystem.core.OutputFactory;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.function.Supplier;
import org.jocl.CL;
import wrapper.core.OpenCLPlatform;
import static cl.core.api.MambaAPIInterface.ImageType.ALL_RAYTRACE_IMAGE;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.RENDER_BUFFER;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CLight;
import cl.core.data.struct.CPath;
import cl.core.data.struct.CRay;
import cl.core.data.struct.CState;
import cl.core.device.RayDeviceMeshRender;
import coordinate.parser.obj.OBJMappedParser;
import coordinate.struct.ByteStruct;
import coordinate.struct.StructByteArray;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

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
    private CAccelerator bvhBuild;
    
    //device priority
    private DeviceType devicePriority = RAYTRACE;
    
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
        this.initBitmap(ALL_RAYTRACE_IMAGE);
        
        //instantiate devices
        deviceRaytrace = new RayDeviceMesh();
        deviceRaytrace.setAPI(this);
        deviceRender = new RayDeviceMeshRender();
        deviceRender.setAPI(this);
        
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
                display.set(RENDER_IMAGE.name(), renderBitmap);
                break;
            case ALL_RAYTRACE_IMAGE:
                raytraceBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y, true);
                overlayBitmap = new BitmapARGB(raytraceImageDimension.x, raytraceImageDimension.y, false);
                overlay = new Overlay(raytraceImageDimension.x, raytraceImageDimension.y);
                renderBitmap = new BitmapARGB(renderImageDimension.x, renderImageDimension.y, false);
                
                display.set(RAYTRACE_IMAGE.name(), raytraceBitmap); 
                display.set(RENDER_IMAGE.name(), renderBitmap);
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
        {
            switch (image) {               
                case ALL_RAYTRACE_IMAGE:
                    deviceRaytrace.readBuffer(IMAGE_BUFFER, buffer-> {
                        this.getBitmap(RAYTRACE_IMAGE).writeColor(buffer.array(), 0, 0, raytraceImageDimension.x, raytraceImageDimension.y);
                        this.display.imageFill(RAYTRACE_IMAGE.name(), this.getBitmap(RAYTRACE_IMAGE));
                    }); 
                    deviceRaytrace.readBuffer(GROUP_BUFFER, buffer-> {
                        overlay.copyToArray(buffer.array());
                    }); break;
                default:
                    break;
            }
        }
        else if(device.equals(RENDER))
        {
            deviceRender.readBuffer(RENDER_BUFFER, buffer-> {
                this.getBitmap(RENDER_IMAGE).writeColor(buffer.array(), 0, 0, renderImageDimension.x, renderImageDimension.y);
                this.display.imageFill(RENDER_IMAGE.name(), this.getBitmap(RENDER_IMAGE));
            });
        }            
    }

    @Override
    public void initMesh(Path path) 
    {
        //load mesh and init mesh variables
        mesh = new CMesh(configurationCL());
        OBJParser parser = new OBJParser();    
        Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh)); //Time parsing
        OutputFactory.print("scene parse time", parseTime.toString());
        mesh.initCLBuffers();
        
        //display material in ui
        controllerImplementation.displaySceneMaterial(parser.getSceneMaterialList());
        
        //build accelerator
        Timer buildTime = Timer.timeThis(() -> {                                   //Time building
            this.bvhBuild = new CPlocBVH(configuration);
            this.bvhBuild.build(mesh);      
        });
        OutputFactory.print("bvh build time", buildTime.toString());
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
        this.deviceRender.set(mesh, bvhBuild);
    }
    
    @Override
    public void initDefaultMesh()
    {
        //A simple cube
        String cube =   "# Cornell Box\n" +
                        "o floor.005\n" +
                        "v 1.014808 -1.001033 -0.985071\n" +
                        "v -0.995168 -1.001033 -0.994857\n" +
                        "v -1.005052 -1.001033 1.035119\n" +
                        "v 0.984925 -1.001033 1.044808\n" +
                        "vn 0.0000 1.0000 -0.0000\n" +
                        "s off\n" +
                        "f 1//1 2//1 3//1 4//1\n" +
                        "o ceiling.005\n" +
                        "v 1.024808 0.988967 -0.985022\n" +
                        "v 1.014924 0.988967 1.044954\n" +
                        "v -1.005052 0.988967 1.035119\n" +
                        "v -0.995168 0.988967 -0.994857\n" +
                        "vn -0.0000 -1.0000 0.0000\n" +
                        "s off\n" +
                        "f 5//2 6//2 7//2 8//2\n" +
                        "o backWall.005\n" +
                        "v 0.984925 -1.001033 1.044808\n" +
                        "v -1.005052 -1.001033 1.035119\n" +
                        "v -1.005052 0.988967 1.035119\n" +
                        "v 1.014924 0.988967 1.044954\n" +
                        "vn 0.0049 -0.0000 -1.0000\n" +
                        "s off\n" +
                        "f 9//3 10//3 11//3 12//3\n" +
                        "o rightWall.005\n" +
                        "v -1.005052 -1.001033 1.035119\n" +
                        "v -0.995168 -1.001033 -0.994857\n" +
                        "v -0.995168 0.988967 -0.994857\n" +
                        "v -1.005052 0.988967 1.035119\n" +
                        "vn 1.0000 0.0000 0.0049\n" +
                        "s off\n" +
                        "f 13//4 14//4 15//4 16//4\n" +
                        "o Small_Box_Short_Box\n" +
                        "v -0.526342 -0.401033 -0.752572\n" +
                        "v -0.699164 -0.401033 -0.173406\n" +
                        "v -0.129998 -0.401033 -0.000633\n" +
                        "v 0.052775 -0.401033 -0.569750\n" +
                        "v 0.052775 -1.001033 -0.569750\n" +
                        "v 0.052775 -0.401033 -0.569750\n" +
                        "v -0.129998 -0.401033 -0.000633\n" +
                        "v -0.129998 -1.001033 -0.000633\n" +
                        "v -0.526342 -1.001033 -0.752572\n" +
                        "v -0.526342 -0.401033 -0.752572\n" +
                        "v 0.052775 -0.401033 -0.569750\n" +
                        "v 0.052775 -1.001033 -0.569750\n" +
                        "v -0.699164 -1.001033 -0.173406\n" +
                        "v -0.699164 -0.401033 -0.173406\n" +
                        "v -0.526342 -0.401033 -0.752572\n" +
                        "v -0.526342 -1.001033 -0.752572\n" +
                        "v -0.129998 -1.001033 -0.000633\n" +
                        "v -0.129998 -0.401033 -0.000633\n" +
                        "v -0.699164 -0.401033 -0.173406\n" +
                        "v -0.699164 -1.001033 -0.173406\n" +
                        "vn 0.0000 1.0000 -0.0000\n" +
                        "vn 0.9521 0.0000 0.3058\n" +
                        "vn 0.3010 -0.0000 -0.9536\n" +
                        "vn -0.2905 0.0000 0.9569\n" +
                        "vn -0.9582 0.0000 -0.2859\n" +
                        "s off\n" +
                        "f 17//5 18//5 19//5 20//5\n" +
                        "f 21//6 22//6 23//6 24//6\n" +
                        "f 25//7 26//7 27//7 28//7\n" +
                        "f 33//8 34//8 35//8 36//8\n" +
                        "f 29//9 30//9 31//9 32//9\n" +
                        "o tall_box\n" +
                        "v 0.530432 0.198967 -0.087419\n" +
                        "v -0.040438 0.198967 0.089804\n" +
                        "v 0.136736 0.198967 0.670674\n" +
                        "v 0.707606 0.198967 0.493451\n" +
                        "v 0.530432 -1.001033 -0.087418\n" +
                        "v 0.530432 0.198967 -0.087419\n" +
                        "v 0.707606 0.198967 0.493451\n" +
                        "v 0.707606 -1.001033 0.493451\n" +
                        "v 0.707606 -1.001033 0.493451\n" +
                        "v 0.707606 0.198967 0.493451\n" +
                        "v 0.136736 0.198967 0.670674\n" +
                        "v 0.136736 -1.001033 0.670674\n" +
                        "v 0.136736 -1.001033 0.670674\n" +
                        "v 0.136736 0.198967 0.670674\n" +
                        "v -0.040438 0.198967 0.089804\n" +
                        "v -0.040438 -1.001033 0.089804\n" +
                        "v -0.040438 -1.001033 0.089804\n" +
                        "v -0.040438 0.198967 0.089804\n" +
                        "v 0.530432 0.198967 -0.087419\n" +
                        "v 0.530432 -1.001033 -0.087418\n" +
                        "vn -0.0000 1.0000 -0.0000\n" +
                        "vn 0.9565 0.0000 -0.2917\n" +
                        "vn 0.2965 0.0000 0.9550\n" +
                        "vn -0.9565 0.0000 0.2917\n" +
                        "vn -0.2965 -0.0000 -0.9550\n" +
                        "s off\n" +
                        "f 37//10 38//10 39//10 40//10\n" +
                        "f 41//11 42//11 43//11 44//11\n" +
                        "f 45//12 46//12 47//12 48//12\n" +
                        "f 49//13 50//13 51//13 52//13\n" +
                        "f 53//14 54//14 55//14 56//14\n" +
                        "o light.005\n" +
                        "v 0.240776 0.978967 -0.158830\n" +
                        "v 0.238926 0.978967 0.221166\n" +
                        "v -0.231068 0.978967 0.218877\n" +
                        "v -0.229218 0.978967 -0.161118\n" +
                        "vn 0.0000 -1.0000 0.0000\n" +
                        "s off\n" +
                        "f 57//15 58//15 59//15 60//15\n" +
                        "o leftWall.000_leftWall.006\n" +
                        "v 1.014808 -1.001033 -0.985071\n" +
                        "v 0.984925 -1.001033 1.044808\n" +
                        "v 1.014924 0.988967 1.044954\n" +
                        "v 1.024808 0.988967 -0.985022\n" +
                        "vn -0.9999 0.0100 -0.0098\n" +
                        "s off\n" +
                        "f 61//16 62//16 63//16 64//16";
        
        //load mesh and init mesh variables
        mesh = new CMesh(configurationCL());   
        OBJParser parser = new OBJParser();        
        parser.readString(cube, mesh);
        mesh.initCLBuffers();
        
        //display material in ui
        controllerImplementation.displaySceneMaterial(parser.getSceneMaterialList());
        
        //build accelerator
        this.bvhBuild = new CPlocBVH(configurationCL());
        this.bvhBuild.build(mesh);    
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
        this.deviceRender.set(mesh, bvhBuild);
    }

    @Override
    public void startDevice(DeviceType device) {
        
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.start();
        else if(device.equals(RENDER))
        { 
            this.deviceRender.start();
        }
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
        if(name == RAYTRACE_IMAGE)
            display.imageFill(RAYTRACE_IMAGE.name(), this.getBitmap(RAYTRACE_IMAGE));
        else if(name == RENDER_IMAGE)
            display.imageFill(RENDER_IMAGE.name(), this.getBitmap(RENDER_IMAGE));
    }

    @Override
    public void setBitmap(ImageType name, BitmapARGB bitmap) {
        switch (name) {
            case RAYTRACE_IMAGE:
                this.raytraceBitmap = bitmap;
            case RENDER_IMAGE:
                this.renderBitmap = bitmap;
            case ALL_RAYTRACE_IMAGE:
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
                return this.deviceRender.isRunning();
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

    @Override
    public boolean isDevicePriority(DeviceType device) {
        return devicePriority.equals(device);
    }

    @Override
    public void setDevicePriority(DeviceType device) {
        this.devicePriority = device;
    }

    @Override
    public DeviceType getDevicePriority() {
        return this.devicePriority;
    }
    
    public CKernel createIntersectionKernel(
            String kernelName, 
            CStructTypeBuffer<CRay> raysBuffer,
            CStructTypeBuffer<CIntersection> isectsBuffer,
            CIntBuffer count,
            CMesh mesh,
            CAccelerator bvhBuild,
            CIntBuffer startNode)
    {
        return configurationCL().program().createKernel(kernelName, raysBuffer, isectsBuffer, count, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvhBuild.getCNodes(), bvhBuild.getCBounds(), startNode);
    }
       
    public CKernel createOcclusionKernel(
            String kernelName, 
            CStructTypeBuffer<CRay> raysBuffer,
            CIntBuffer hits,
            CIntBuffer count,
            CMesh mesh,
            CNormalBVH bvhBuild)
    { 
        return configurationCL().program().createKernel(kernelName, raysBuffer, hits, count, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvhBuild.getCNodes(), bvhBuild.getCBounds());
    } 
    
    public CKernel sampleLightsKernel(
            String kernelName,
            CStructTypeBuffer<CPath> lightPaths,
            CStructTypeBuffer<CLight> lights,
            CIntBuffer totalLights,            
            CIntBuffer count,
            CStructTypeBuffer<CState> state,
            CMesh mesh
    )
    {     
        
        return configurationCL().program().createKernel(kernelName, lightPaths, lights, totalLights, count, state, mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize());
    }
    
    public CKernel createKernel(String kernelName, CMemory... buffers)
    {
        return configurationCL().program().createKernel(kernelName, buffers);
    }
    
    public CIntBuffer allocIntValue(String name, int value, long flag)
    {
        return CBufferFactory.initIntValue(name, configurationCL().context(), configurationCL().queue(), value, flag);
    }
    
    public CFloatBuffer allocFloatValue(String name, float value, long flag)
    {
        return CBufferFactory.initFloatValue(name, configurationCL().context(), configurationCL().queue(), value, flag);
    }
    
    public <B extends ByteStruct> CStructTypeBuffer<B> allocStructType(String name, Class<B> clazz, int size, long flag)
    {
        return CBufferFactory.allocStructType(name, configurationCL().context(), clazz, size, flag);
    }
    
    public <B extends ByteStruct> CStructTypeBuffer<B> allocStructType(String name, StructByteArray structArray, int size, long flag)
    {
        return CBufferFactory.allocStructType(name, configurationCL().context(), structArray, flag);
    }
    
    public CIntBuffer allocInt(String name, int size, long flag)
    {
        return CBufferFactory.allocInt(name, configurationCL().context(), size, flag);
    }
    
    public CFloatBuffer allocFloat(String name, int size, long flag)
    {
        return CBufferFactory.allocFloat(name, configurationCL().context(), size, flag);
    }
    
    public void execute1D(CKernel kernel, int globalSize, int localSize)
    {
        configurationCL().queue().put1DRangeKernel(kernel, globalSize, localSize);
    }
}
