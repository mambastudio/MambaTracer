/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package raytrace;

import bitmap.core.AbstractDisplay;
import cl.ui.fx.BlendDisplay;
import bitmap.image.BitmapRGBE;
import cl.abstracts.MambaAPIInterface;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.abstracts.MambaAPIInterface.ImageType.ALL_RAYTRACE_IMAGE;
import cl.abstracts.RayDeviceInterface;
import cl.algorithms.CEnvironment;
import cl.data.CPoint2;
import cl.data.CPoint3;
import cl.data.CVector3;
import cl.fx.UtilityHandler;
import cl.scene.CMesh;
import cl.scene.CNormalBVH;
import cl.struct.CBound;
import cl.struct.CMaterial2;
import cl.ui.fx.material.MaterialFX2;
import coordinate.parser.obj.OBJInfo;
import coordinate.parser.obj.OBJMappedParser;
import coordinate.parser.obj.OBJParser;
import coordinate.utility.Timer;
import coordinate.utility.Value2Di;
import java.nio.file.Path;
import org.jocl.CL;
import raytrace.cl.RaytraceSource;
import wrapper.core.CMemory;
import wrapper.core.OpenCLConfiguration;

/**
 *
 * @author user
 */
public class RaytraceAPI implements MambaAPIInterface<MaterialFX2, RaytraceUIController>{
    //Opencl configuration for running single ray tracing program 
    private OpenCLConfiguration configuration = null;
    
    //The display to be used, is independent of ui since it can be file writer (jpg, png,...) display
    private BlendDisplay displayRT = null;
    
    //Controller 
    private RaytraceUIController controllerImplementation = null;
      
    //Images dimension
    private Value2Di raytraceImageDimension;
    
    //ray casting device
    private RaytraceDevice deviceRaytrace;
    
    //device priority
    private DeviceType devicePriority = RAYTRACE;
    
    //mesh and accelerator
    private CMesh mesh;
    private CNormalBVH bvhBuild;
    
    //material for editing
    private MaterialFX2[] matFXArray;
    
    //environment map
    private RaytraceEnvironment envmap;
    
    public RaytraceAPI()
    {
        CL.setExceptionsEnabled(true);
         
        //opencl configuration
        this.initOpenCLConfiguration();
    }
    
    @Override
    public void initOpenCLConfiguration() {
        if(configuration != null) 
            return;
        configuration = OpenCLConfiguration.getDefault(RaytraceSource.readFiles());
    }

    @Override
    public OpenCLConfiguration getConfigurationCL() {
        return configuration;
    }

    @Override
    public void init() {
        //dimension of images
        this.raytraceImageDimension = new Value2Di(500, 500);
        
        //create bitmap images
        this.initBitmap(ALL_RAYTRACE_IMAGE);
        
        //instantiate devices
        deviceRaytrace = new RaytraceDevice(
                this.raytraceImageDimension.x, 
                this.raytraceImageDimension.y);        
                
        //init default mesh before api
        initDefaultMesh();
        
        
         //envmap
        envmap = new RaytraceEnvironment(configuration);
        
        //set api       
        deviceRaytrace.setAPI(this);
        
       
                        
        //start ray tracing
        deviceRaytrace.start();
    }

    @Override
    public <D extends AbstractDisplay> D getDisplay(Class<D> displayClass) {
        if(BlendDisplay.class.isAssignableFrom(displayClass))        
            return (D) displayRT;
        else
            throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public <D extends AbstractDisplay> void setDisplay(Class<D> displayClass, D display) {
        if(BlendDisplay.class.isAssignableFrom(displayClass))        
            displayRT = (BlendDisplay) display;
        else
            throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Value2Di getImageSize(ImageType imageType) {
        switch (imageType) {            
            case RAYTRACE_IMAGE:
                return this.raytraceImageDimension;
            default:
                return null;
        }
    }

    @Override
    public void setImageSize(ImageType imageType, int width, int height) {
        switch (imageType) {            
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
            default:
                return -1;
        }
    }

    @Override
    public void render(DeviceType device) {
        if(device.equals(RAYTRACE))
            deviceRaytrace.start();
    }
    
    @Override
    public void initMesh(Path path) {
        //load mesh and init mesh variables        
        OBJMappedParser parser = new OBJMappedParser();        
        parser.readAttributes(path.toUri());
        
        boolean succeed = UtilityHandler.runJavaFXThread(()->{
            //init size (estimate) of coordinate list/array
            OBJInfo info = parser.getInfo();
            return controllerImplementation.showOBJStatistics(info);
        });
        if(succeed)
        {
            mesh = null;
            mesh = new CMesh(getConfigurationCL());
            
            //init array sizes (eventually they will grow if bounds reach)
            mesh.initCoordList(CPoint3.class, CVector3.class, CPoint2.class, 0, 0, 0, 0);

            Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh)); //Time parsing
            //UI.print("parse-time", parseTime.toString()); 
            mesh.initCLBuffers();
            setupMaterial();

            //display material in ui
            //controllerImplementation.displaySceneMaterial(parser.getSceneMaterialList());

            //build accelerator
            Timer buildTime = Timer.timeThis(() -> {                                   //Time building
                this.setMessage("building accelerator");
                this.bvhBuild = new CNormalBVH(configuration);
                this.bvhBuild.build(mesh);      
            });
           // UI.print("build-time", buildTime.toString());

            //set to device for rendering/raytracing
            this.deviceRaytrace.set(mesh, bvhBuild);
            
            //init various buffers and kernels to reflect on new model            
            deviceRaytrace.setAPI(this);
            
            this.repositionCameraToSceneRT();
        }
    }

    
    @Override
    public void initDefaultMesh() {
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
        CMemory<CMaterial2> materialsc =  mesh.clMaterials();
        
        CMaterial2 emitterc = materialsc.get(6);
        //emitter.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki            
        emitterc.setEmitter(1f, 1f, 1f);
               
        CMaterial2 rightc = materialsc.get(3);           
        rightc.setDiffuse(0, 0.8f, 0);

        CMaterial2 leftc = materialsc.get(7);         
        leftc.setDiffuse(0.8f, 0f, 0);

        CMaterial2 backc = materialsc.get(2);           
        backc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial2 ceilingc = materialsc.get(1);           
        ceilingc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial2 floorc = materialsc.get(0);           
        floorc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki

        CMaterial2 smallboxc = materialsc.get(4);  
        smallboxc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
        //smallbox.setEmitter(1, 1, 1);
        //smallbox.setEmitterEnabled(true);

        CMaterial2 tallboxc = materialsc.get(5);           
        tallboxc.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki  
        
        //transfer material to device
        materialsc.transferToDevice();
        
                
        //build accelerator
        bvhBuild = new CNormalBVH(configuration);
        bvhBuild.build(mesh);  
        
        //set to device for rendering/raytracing
        this.deviceRaytrace.set(mesh, bvhBuild);
        
        //set up material for modification
        setupMaterial();
    }
    
    private void setupMaterial()
    {
        int matSize = mesh.clMaterials().getSize();
        
        matFXArray = new MaterialFX2[matSize];
        for(int i = 0; i<matFXArray.length; i++)
        {            
            matFXArray[i] = new MaterialFX2();
            matFXArray[i].setCMaterial(mesh.clMaterials().get(i));
        }
    }

    @Override
    public void startDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.start();
        else
            System.out.println("no device");
    }

    @Override
    public void pauseDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.pause();
        else
            System.out.println("no device");
    }

    @Override
    public void stopDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.stop();
        else
            System.out.println("no device");
    }

    @Override
    public void resumeDevice(DeviceType device) {
        if(device.equals(RAYTRACE))
            this.deviceRaytrace.resume();
        else
            System.out.println("no device");
    }

    @Override
    public boolean isDeviceRunning(DeviceType device) {
        switch (device) {
            case RAYTRACE:
                return this.deviceRaytrace.isRunning();           
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
        if(device.equals(RAYTRACE))
            return deviceRaytrace;
        return null;
    }
    
    public <Device extends RayDeviceInterface> Device getDevice(Class<Device> deviceClass)
    {
        if(RaytraceDevice.class.isAssignableFrom(deviceClass))        
            return (Device) deviceRaytrace;
        else
            throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void set(DeviceType device, RayDeviceInterface deviceImplementation) {
        if(device.equals(RAYTRACE))
        {
            this.deviceRaytrace = (RaytraceDevice) deviceImplementation;
            this.deviceRaytrace.setAPI(this);
        }        
    }

    @Override
    public RaytraceUIController getController(String controller) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void set(String controller, RaytraceUIController controllerImplementation) {
        this.controllerImplementation = controllerImplementation;        
        this.controllerImplementation.setAPI(this);
    }

    @Override
    public void setMaterial(int index, MaterialFX2 material) {
        //TODO 
        matFXArray[index].setMaterial(material);
    }

    @Override
    public MaterialFX2 getMaterial(int index) {
        return matFXArray[index];
    }

    @Override
    public void setEnvironmentMap(BitmapRGBE bitmap) {
        envmap.setEnvironmentMap(bitmap);        
        //deviceRaytrace.setEnvMapInKernel();               
        deviceRaytrace.resume();
    }

    public void setIsEnvmapPresent(boolean value)
    {
        envmap.setIsPresent(value);
    }
    
    
    public RaytraceEnvironment getEnvironmentalMapCL()
    {
        return envmap;
    }
    
    //this changes direction (by reseting direction to z axis)
    public void repositionCameraToSceneRT()
    {
        RaytraceDevice device = getDevice(RaytraceDevice.class);
        CBound bound = device.getBound();
        device.setPriorityBound(bound);
        device.getCameraModel().set(bound.getCenter().add(new CVector3(0, 0, -3)), new CPoint3(), new CVector3(0, 1, 0), 45);
        device.reposition(bound);
    }
   
    
    //this changes only position but direction remains intact
    public void repositionCameraToBoundRT(CBound bound)
    {
        RaytraceDevice device = getDevice(RaytraceDevice.class);
        device.setPriorityBound(bound);
        device.reposition(bound);
    }
}
