/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.main;

import cl.ui.fx.material.MaterialFX;
import cl.device.CDeviceRT;
import cl.device.CDeviceGI;
import cl.scene.CNormalBVH;
import cl.scene.CMesh;
import cl.struct.CMaterial;
import bitmap.core.AbstractDisplay;
import bitmap.display.BlendDisplay;
import bitmap.display.ImageDisplay;
import cl.abstracts.MambaAPIInterface;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.abstracts.MambaAPIInterface.DeviceType.RENDER;
import static cl.abstracts.MambaAPIInterface.ImageType.ALL_RAYTRACE_IMAGE;
import cl.abstracts.RayDeviceInterface;
import cl.algorithms.CEnvMap;
import cl.data.CPoint2;
import cl.data.CPoint3;
import cl.data.CVector3;
import coordinate.parser.obj.OBJParser;
import coordinate.utility.Value2Di;
import java.nio.file.Path;
import org.jocl.CL;
import cl.kernel.CSource;
import coordinate.parser.obj.OBJInfo;
import coordinate.parser.obj.OBJMappedParser;
import coordinate.utility.Timer;
import wrapper.core.CMemory;
import wrapper.core.OpenCLConfiguration;

/**
 *
 * @author user
 */
public class TracerAPI implements MambaAPIInterface<AbstractDisplay, MaterialFX, UserInterfaceFXMLController>  {

    //Opencl configuration for running single ray tracing program (might add one for future material editor)
    private OpenCLConfiguration configuration = null;
    
    //The display to be used, is independent of ui since it can be file writer (jpg, png,...) display
    private BlendDisplay displayRT = null;
    private ImageDisplay displayGI = null;
    
    //Controller 
    private UserInterfaceFXMLController controllerImplementation = null;
      
    //Images dimension
    private Value2Di raytraceImageDimension;
    private Value2Di renderImageDimension;
    
    //ray casting device
    private CDeviceRT deviceRaytrace;
    private CDeviceGI deviceRender;   
    
    //device priority
    private DeviceType devicePriority = RAYTRACE;
    
    //mesh and accelerator
    private CMesh mesh;
    private CNormalBVH bvhBuild;
    
    //material for editing
    private MaterialFX[] matFXArray;
    
    //environment map
    private CEnvMap envmap;
    
    public TracerAPI()
    {
        CL.setExceptionsEnabled(true);
         
        //opencl configuration
        this.initOpenCLConfiguration();
    }
    
    @Override
    public final void initOpenCLConfiguration() {
        if(configuration != null) 
            return;
        configuration = OpenCLConfiguration.getDefault(CSource.readFiles());  
    }

    @Override
    public final OpenCLConfiguration getConfigurationCL() {
        return configuration;
    }

    //This is not called in the constructure but rather in the main method
    @Override
    public void init() {
        
        //dimension of images
        this.raytraceImageDimension = new Value2Di(500, 500);
        this.renderImageDimension = new Value2Di(500, 500);
        
        //create bitmap images
        this.initBitmap(ALL_RAYTRACE_IMAGE);
        
        //instantiate devices
        deviceRaytrace = new CDeviceRT(
                this.raytraceImageDimension.x, 
                this.raytraceImageDimension.y);        
        deviceRender = new CDeviceGI(
                this.renderImageDimension.x,
                this.renderImageDimension.y);
        
        //init default mesh before api
        this.initDefaultMesh();
        
         //envmap
        envmap = new CEnvMap(configuration);
        
        //set api
        deviceRender.setAPI(this);
        deviceRaytrace.setAPI(this);
        
       
                        
        //start ray tracing
        deviceRaytrace.start();
        
    }

    public BlendDisplay getBlendDisplayRT()
    {
        return (BlendDisplay) this.getBlendDisplay(RAYTRACE);
    }
    
    public ImageDisplay getBlendDisplayGI()
    {
        return (ImageDisplay) this.getBlendDisplay(RENDER);
    }

    
    @Override
    public AbstractDisplay getBlendDisplay(DeviceType type) {
        switch (type) 
        {
            case RAYTRACE:
                return displayRT;
            case RENDER:
                return displayGI;
            default:
                return null;
        }
    }

    @Override
    public void setBlendDisplay(DeviceType type, AbstractDisplay display) {
        switch (type) 
        {
            case RAYTRACE:
                displayRT = (BlendDisplay) display;
                break;
            case RENDER:
                displayGI = (ImageDisplay) display;
                break;
            default:
                break;
        }
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
    public void setImageSize(ImageType imageType, int width, int height) {
        switch (imageType) {
            case RENDER_IMAGE:
                this.renderImageDimension.set(width, height);                
            case RAYTRACE_IMAGE:
                this.raytraceImageDimension.set(width, height);
            default:
                break;
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
    public void render(DeviceType device) {
        if(device.equals(RAYTRACE))
            deviceRaytrace.start();
        else if(device.equals(RENDER))
            deviceRender.start();
    }
    
    @Override
    public void initDefaultMesh()
    {
        //A simple cornell box
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
        mesh = new CMesh(configuration);   
        OBJParser parser = new OBJParser();        
        parser.readString(cube, mesh);
        mesh.initCLBuffers();
        
        //modify materials
        /*
            0 floor
            1 ceiling
            2 back wall
            3 right wall
            4 small box
            5 tall box
            6 emitter
            7 left wall            
        */
        
        
        //NEW MATERIAL
        CMemory<CMaterial> materialsc =  mesh.clMaterials();
        
        CMaterial emitterc = materialsc.get(6);
        //emitter.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki            
        emitterc.setEmitter(1f, 1f, 1f);
        emitterc.setEmitterPower(20);
        emitterc.setEmitterEnabled(true);

        CMaterial rightc = materialsc.get(3);           
        rightc.setDiffuse(0, 0.8f, 0);

        CMaterial leftc = materialsc.get(7);         
        leftc.setDiffuse(0.8f, 0f, 0);

        CMaterial backc = materialsc.get(2);           
        backc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial ceilingc = materialsc.get(1);           
        ceilingc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial floorc = materialsc.get(0);           
        floorc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial smallboxc = materialsc.get(4);  
        smallboxc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
        //smallbox.setEmitter(1, 1, 1);
        //smallbox.setEmitterEnabled(true);

        CMaterial tallboxc = materialsc.get(5);           
        tallboxc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki  
        
        //transfer material to device
        materialsc.transferToDevice();
        
                
        //build accelerator
        bvhBuild = new CNormalBVH(configuration);
        bvhBuild.build(mesh);  
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
        this.deviceRender.set(mesh, bvhBuild);
        
        //set up material for modification
        setupMaterial();
        
        //Set cameraModel new position
        //reposition(mesh.getBound());
        //updateCamera();
    }

    //TODO: Make sure the output is displayed in the most adequate console
    @Override
    public void initMesh(Path path) {
        //load mesh and init mesh variables
        mesh = new CMesh(getConfigurationCL());
        OBJMappedParser parser = new OBJMappedParser();        
        parser.readAttributes(path.toUri());
        
        //init size (estimate) of coordinate list/array
        OBJInfo info = parser.getInfo();
        controllerImplementation.showOBJStatistics(info);
        
        //init array sizes (eventually they will grow if bounds reach)
        mesh.initCoordList(CPoint3.class, CVector3.class, CPoint2.class, 0, 0, 0, 0);
        
        Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh)); //Time parsing
        //UI.print("parse-time", parseTime.toString()); 
        mesh.initCLBuffers();
        setupMaterial();
        
        //display material in ui
//        controllerImplementation.displaySceneMaterial(parser.getSceneMaterialList());
        
        //build accelerator
        Timer buildTime = Timer.timeThis(() -> {                                   //Time building
            this.setMessage("building accelerator");
            this.bvhBuild = new CNormalBVH(configuration);
            this.bvhBuild.build(mesh);      
        });
       // UI.print("build-time", buildTime.toString());
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
        this.deviceRender.set(mesh, bvhBuild);
                
        //init various buffers and kernels to reflect on new model
        deviceRender.setAPI(this);
        deviceRaytrace.setAPI(this);
    }
    
    private void setupMaterial()
    {
        int matSize = mesh.clMaterials().getSize();
        
        matFXArray = new MaterialFX[matSize];
        for(int i = 0; i<matFXArray.length; i++)
        {            
            matFXArray[i] = new MaterialFX();
            matFXArray[i].setCMaterial(mesh.clMaterials().get(i));
        }
    }
    
    public void setMaterial(int index, MaterialFX matFX)
    {                
        CMemory<CMaterial> materialsc = mesh.clMaterials();          
        matFXArray[index].setMaterial(matFX);
        materialsc.transferToDevice();        
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
    public void setDevicePriority(DeviceType device) {
        this.devicePriority = device;
    }

    @Override
    public DeviceType getDevicePriority() {
        return this.devicePriority;
    }

    @Override
    public boolean isDevicePriority(DeviceType device) {
        return devicePriority.equals(device);
    }

    @Override
    public RayDeviceInterface getDevice(DeviceType device) {
        if(device.equals(RENDER))
            return deviceRender;
        else if(device.equals(RAYTRACE))
            return deviceRaytrace;
        return null;
    }
    
    public CDeviceRT getDeviceRT()
    {
        return (CDeviceRT) getDevice(RAYTRACE);
    }
    
    public CDeviceGI getDeviceGI()
    {
        return (CDeviceGI) getDevice(RENDER);
    }

    @Override
    public void set(DeviceType device, RayDeviceInterface deviceImplementation) {
        if(device.equals(RAYTRACE))
        {
            this.deviceRaytrace = (CDeviceRT) deviceImplementation;
            this.deviceRaytrace.setAPI(this);
        }
        else if(device.equals(RENDER))
        {
            this.deviceRender = (CDeviceGI) deviceImplementation;
            this.deviceRender.setAPI(this);
        }
    }

    @Override
    public UserInterfaceFXMLController getController(String controller) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void set(String controller, UserInterfaceFXMLController controllerImplementation) {
        this.controllerImplementation = controllerImplementation;
        this.controllerImplementation.setAPI(this);
    }

    @Override
    public void set(int index, MaterialFX material) {
        //TODO 
        matFXArray[index].setMaterial(material);
    }

    @Override
    public MaterialFX get(int index) {
        return matFXArray[index];
    }

    @Override
    public void setEnvironmentMap(float[] rgb4, int width, int height) {
        envmap.setEnvironmentMap(rgb4, width, height);
        deviceRaytrace.setEnvMapInKernel();
        deviceRaytrace.resume();
    }

    public void setIsEnvmapPresent(boolean value)
    {
        envmap.setIsPresent(value);
    }
    
    
    public CEnvMap getEnvironmentalMapCL()
    {
        return envmap;
    }
    
}
