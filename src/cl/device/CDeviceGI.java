/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.device;

import cl.struct.CTextureData;
import cl.algorithms.CCompact;
import cl.struct.CState;
import cl.struct.CPath;
import cl.struct.CCameraModel;
import cl.struct.CCamera;
import cl.struct.CIntersection;
import cl.struct.CRay;
import cl.struct.CBound;
import bitmap.display.ImageDisplay;
import bitmap.image.BitmapARGB;
import static cl.abstracts.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.abstracts.RayDeviceInterface;
import cl.algorithms.CEnvironment;
import cl.algorithms.CImage;
import cl.algorithms.CLightConfiguration;
import cl.data.CPoint2;
import cl.scene.CMesh;
import cl.scene.CNormalBVH;
import cl.algorithms.CTextureApplyPass;
import cl.struct.CMaterial2;
import cl.ui.fx.main.TracerAPI;
import java.math.BigInteger;
import java.util.Random;
import thread.model.LambdaThread;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.FloatValue;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CDeviceGI implements RayDeviceInterface<
        TracerAPI, 
        ImageDisplay, 
        CMaterial2, 
        CMesh,
        CNormalBVH,
        CBound,
        CCameraModel, 
        CCamera> {
    
    //render thread
    LambdaThread renderThread = new LambdaThread();
    
    CMesh mesh = null;
    CNormalBVH bvh = null;
    
    int width, height;
    BitmapARGB renderBitmap;
    
    //global and local size
    int globalWorkSize, localWorkSize;
    
    boolean isImageVisible = true;
    
    //CL
    private CMemory<CState> gStateBuffer;
    CMemory<IntValue> gLength = null; //frame w * h
    CMemory<IntValue> gCountBuffer = null;
    CMemory<IntValue> gPixelIndicesBuffer = null;
    CMemory<CCamera> gCamera = null;    
    CMemory<CRay> gRaysBuffer = null;
    CMemory<CRay> gOcclusRaysBuffer = null;
    CMemory<CIntersection> gIsectBuffer = null;
    CMemory<CPath> gBPathBuffer = null;
    CTextureApplyPass texApplyPass = null;
    CMemory<CTextureData> texBuffer = null;
    CCompact compactHybrid;
    
    private CEnvironment envmap = null;
    private CImage image = null;
  
    //kernels   
    private CKernel gInitCameraRaysKernel;
    private CKernel gInitIsectsKernel;
    private CKernel gInitPathsKernel;
    private CKernel gInitPixelIndicesKernel;
    private CKernel gIntersectPrimitivesKernel; 
    private CKernel gSetupBSDFKernel;
    private CKernel gLightHitPassKernel;  
    private CKernel gSampleBSDFRayDirectionKernel;
    private CKernel gDirectLightKernel;
    private CKernel gTextureInitPassKernel;
    private CKernel gUpdateToTextureColorGIKernel;
    
    
    private TracerAPI api;
    
    
    OpenCLConfiguration configuration;
    ImageDisplay display;
    
    CLightConfiguration lightConfiguration;
    
    int i = 0;
    
    
    public CDeviceGI(int w, int h)
    {
        this.width = w; 
        this.height = h;
    }
    
    public final void setup(int w, int h)
    {
        this.width = w;
        this.height = h;
        this.renderBitmap = new BitmapARGB(width, height);
        this.globalWorkSize = w * h;
        this.localWorkSize = 100;
        
        createBuffers();
        createKernels();
        compactHybrid.init(gIsectBuffer, gPixelIndicesBuffer, gCountBuffer);
    }
        
    
    public void createBuffers()
    {
        //clear references to make new ones
        //if you don't do this, when dealing with large arrays, there is some data corruption. I don't know why
        image = null;
        gStateBuffer         = null;
        gPixelIndicesBuffer  = null;
        gCountBuffer         = null;
        gLength              = null;
        gRaysBuffer          = null;
        gOcclusRaysBuffer    = null;
        gCamera              = null;
        gBPathBuffer         = null;
        gIsectBuffer         = null;
        compactHybrid        = null;
        texBuffer            = null;
        texApplyPass         = null;
        
        
        //start assigning
        image = new CImage(configuration, width, height);
        
        gStateBuffer         = configuration.createBufferB(CState.class, 1, READ_WRITE);
        gPixelIndicesBuffer  = configuration.createBufferI(IntValue.class, globalWorkSize, READ_WRITE);
        gCountBuffer         = configuration.createFromI(IntValue.class, new int[]{globalWorkSize}, READ_WRITE);
        gLength              = configuration.createFromI(IntValue.class, new int[]{globalWorkSize}, READ_WRITE);
        gRaysBuffer          = configuration.createBufferB(CRay.class, globalWorkSize, READ_WRITE);
        gOcclusRaysBuffer    = configuration.createBufferB(CRay.class, globalWorkSize, READ_WRITE);   
        gCamera              = configuration.createBufferB(CCamera.class, 1, READ_WRITE);
        gBPathBuffer         = configuration.createBufferB(CPath.class, globalWorkSize, READ_WRITE);
        gIsectBuffer         = configuration.createBufferB(CIntersection.class, globalWorkSize, READ_WRITE);
        compactHybrid        = new CCompact(configuration);       
        texBuffer            = configuration.createBufferI(CTextureData.class, globalWorkSize, READ_WRITE);
        texApplyPass         = new CTextureApplyPass(api, texBuffer, gCountBuffer);
    }
    
    public void createKernels()
    {
        gInitCameraRaysKernel           = configuration.createKernel("InitCameraRayDataJitter", gCamera, gRaysBuffer, gStateBuffer, gLength);
        gInitPathsKernel                = configuration.createKernel("InitPathData", gBPathBuffer, gLength);
        gInitIsectsKernel               = configuration.createKernel("InitIntersection", gIsectBuffer, gLength);
        gInitPixelIndicesKernel         = configuration.createKernel("InitIntDataToIndex", gPixelIndicesBuffer, gLength);
        gIntersectPrimitivesKernel      = configuration.createKernel("IntersectPrimitives", gRaysBuffer, gIsectBuffer, gCountBuffer, mesh.clPoints(), mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getNodes(), bvh.getBounds());
        gSetupBSDFKernel                = configuration.createKernel("SetupBSDFPath", gIsectBuffer, gRaysBuffer, gBPathBuffer, mesh.clMaterials(), gPixelIndicesBuffer, gCountBuffer);
        gLightHitPassKernel             = configuration.createKernel("LightHitPass", gIsectBuffer, gRaysBuffer, gBPathBuffer,  lightConfiguration.getCLightCount(), mesh.clMaterials(), mesh.clPoints(), mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), image.getFrameAccum(), gPixelIndicesBuffer, gCountBuffer, envmap.getRgbCL(), envmap.getEnvLumCL(), envmap.getEnvLumSATCL(), envmap.getLightGrid());
        gSampleBSDFRayDirectionKernel   = configuration.createKernel("SampleBSDFRayDirection", gIsectBuffer, gRaysBuffer, gBPathBuffer, mesh.clMaterials(), gPixelIndicesBuffer, gStateBuffer, gCountBuffer);
        gDirectLightKernel              = configuration.createKernel("DirectLight", gBPathBuffer, gIsectBuffer, lightConfiguration.getCLightInfoList(), lightConfiguration.getCLightCount(), gOcclusRaysBuffer, image.getFrameAccum(), gPixelIndicesBuffer, gCountBuffer, gStateBuffer, mesh.clMaterials(), mesh.clPoints(), mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getNodes(), bvh.getBounds(), envmap.getRgbCL(), envmap.getEnvLumCL(), envmap.getEnvLumSATCL(), envmap.getLightGrid());
        gTextureInitPassKernel          = configuration.createKernel("texturePassGI", gBPathBuffer, gIsectBuffer, texBuffer, gPixelIndicesBuffer, gCountBuffer);
        gUpdateToTextureColorGIKernel   = configuration.createKernel("updateToTextureColorGI", gBPathBuffer, texBuffer, gPixelIndicesBuffer, gCountBuffer);
      }
    
    public void prepareStart()
    {
        this.image.initBuffers();
        this.initLight();
        this.initFrameCount();
        this.updateCamera();
    }
    
    @Override
    public void start()
    {
        //set screen black first       
        setScreenBlack(true);
        
        //start setting up for rendering
        this.prepareStart();
        
        i = 0;
        
        renderThread.startExecution(()-> {
            execute();
            
        });      
    }
    
    public void setScreenBlack(boolean isBlack)
    {
        this.renderBitmap.reset(isBlack);
        this.display.imageFill(renderBitmap);
        
        this.isImageVisible = isBlack;
    }
    
    private void loop()
    {
        renderThread.chill();
        
        //before we proceed, set new seed state for random number generation
        allocateState(); 
        configuration.execute1DKernel(gInitCameraRaysKernel, globalWorkSize, localWorkSize);
        configuration.execute1DKernel(gInitPathsKernel, globalWorkSize, localWorkSize); 
        configuration.execute1DKernel(gInitIsectsKernel, globalWorkSize, localWorkSize);  
        configuration.execute1DKernel(gInitPixelIndicesKernel, globalWorkSize, localWorkSize);
        //reset intersection count
        resetIntersectionCount();
        for(int pathLength = 1; pathLength<=15; pathLength++)
        {   
            
            //do actual intersection
            configuration.execute1DKernel(gIntersectPrimitivesKernel, globalWorkSize, localWorkSize);
            
            //before we proceed, set new seed state for random number generation
            allocateState(); 
            
            //set up bsdf and deal with implicit light hit (update hit to accum buffer)
            configuration.execute1DKernel(gSetupBSDFKernel, globalWorkSize, localWorkSize);
            configuration.execute1DKernel(gLightHitPassKernel, globalWorkSize, localWorkSize);  
            
            //compact intersections (light hits are not factored in)
            compactHybrid.execute();  
                      
            //pass texture
            configuration.execute1DKernel(gTextureInitPassKernel, globalWorkSize, localWorkSize);
            texApplyPass.process();
            configuration.execute1DKernel(gUpdateToTextureColorGIKernel, globalWorkSize, localWorkSize);
        
            directLightEvaluation();                      
            configuration.execute1DKernel(gSampleBSDFRayDirectionKernel, globalWorkSize, localWorkSize);
            
            
            
            //normal, opacity, colormap
            if(pathLength == 1)
            {
                
            }
           
        }
        //configuration.execute1DKernel(gUpdateImageKernel, globalWorkSize, localWorkSize);
        image.processImage();
        outputImage();
        
        if(i < 5)
            this.envmap.adaptiveUpdate();
        
        renderThread.chill();
        
        i++;
        
        //renderThread.pauseExecution();
    }
    
    public void directLightEvaluation()
    {
        allocateState();//set new seed state        
        configuration.execute1DKernel(gDirectLightKernel, globalWorkSize, localWorkSize);
    }
    
    //called by the UI
    public void clearImage()
    {
        this.renderBitmap.reset(false);
        this.display.imageFill(RENDER_IMAGE.name(), renderBitmap);
        this.isImageVisible = true;
    }
    
    public boolean isImageCleared()
    {
        return this.isImageVisible;
    }
    
    //currently mesh only
    public void initLight()
    {
        //set camera position in light grid        
        envmap.setCameraPosition(getCameraModel().getCameraStruct().position);
        
        //init light configuration (tree, and env map)
        lightConfiguration.initLights(mesh, envmap);
        
        gLightHitPassKernel.putArg(3, lightConfiguration.getCLightCount());
        gDirectLightKernel.resetPutArgs(gBPathBuffer, gIsectBuffer, lightConfiguration.getCLightInfoList(), lightConfiguration.getCLightCount(), gOcclusRaysBuffer, image.getFrameAccum(), gPixelIndicesBuffer, gCountBuffer, gStateBuffer, 
                mesh.clMaterials(), mesh.clPoints(), mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getNodes(), bvh.getBounds()); 
       
        System.out.println("light area count : " +lightConfiguration.getAreaLightCount());
    }
    
    @Override
    public void updateCamera(){
        CCamera cam = api.getDeviceRT().getCameraModel().getCameraStruct();
        cam.setDimension(new CPoint2(width, height));  
        gCamera.setCL(cam);
    }
    
    private void resetIntersectionCount()
    {
        gCountBuffer.setCL(new IntValue(globalWorkSize));
    }
    
    private void incrementFrameCount()
    {
        image.incrementFrameCount();
    }
    
    private void initFrameCount()
    {
        image.getFrameCount().setCL(new FloatValue(1));
    }  
    
    public void setGamma(float value)
    {
        image.setGamma(value);
        image.processImage();
        outputImage();
    }
    
    public float getGamma()
    {
        return image.getGamma();        
    }
    
    public void setExposure(float value)
    {
        image.setExposure(value);
        image.processImage();
        outputImage();
    }
    
    public float getExposure()
    {
        return image.getExposure();        
    }
   
    @Override
    public void outputImage() {      
        //transfer data from opencl to cpu
        image.getFrameARGB().transferFromDevice();
        //write to bitmap
        renderBitmap.writeColor((int[]) image.getFrameARGB().getBufferArray(), 0, 0, width, height);
        //image fill
        display.imageFill(renderBitmap);
    }
    
    private void allocateState()
    {
        //init seed, image dimension and increment frame count
        CState state = new CState();
            
        //seed for current frame count
        int seed0 = BigInteger.probablePrime(30, new Random()).intValue();
        int seed1 = BigInteger.probablePrime(30, new Random()).intValue();

        state.setSeed(seed0, seed1);                       
        state.setFrameCount(image.getFrameCount().getCL().v);
        gStateBuffer.setCL(state);
    }
        
    @Override
    public void setAPI(TracerAPI api) {
        this.api = api;        
        init(api.getConfigurationCL(), api.getDisplay(ImageDisplay.class));        
    }
    
    private void init(OpenCLConfiguration platform, ImageDisplay display)
    {
        this.configuration = platform;
        this.display = display;
        this.envmap = api.getEnvironmentalMapCL();
        this.lightConfiguration = new CLightConfiguration(configuration);
        
        setup(width, height);
    }
    
    public void setEnvMapInKernel()
    {
        gLightHitPassKernel.putArg(13, envmap.getRgbCL());
        gLightHitPassKernel.putArg(14, envmap.getEnvLumCL());
        gLightHitPassKernel.putArg(15, envmap.getEnvLumSATCL());
        
        gDirectLightKernel.putArg(17, envmap.getRgbCL());
        gDirectLightKernel.putArg(18, envmap.getEnvLumCL());
        gDirectLightKernel.putArg(19, envmap.getEnvLumSATCL());
    }

    @Override
    public void set(CMesh mesh, CNormalBVH bvhBuild) {
        this.mesh = null;
        this.bvh = null;
        
        this.mesh = mesh;
        this.bvh = bvhBuild;
    }

    @Override
    public void setGlobalSize(int globalSize) {
        this.globalWorkSize = globalSize;
    }

    @Override
    public void setLocalSize(int localSize) {
        this.localWorkSize = localSize;
    }

    @Override
    public void execute() {
        //execute pause             
        loop();      
        configuration.finish();
        incrementFrameCount();
        //renderThread.pauseExecution(); 
    }
    @Override
    public void pause() {
        renderThread.pauseExecution();
    }

    @Override
    public void stop() {
        renderThread.stopExecution();
    }

    @Override
    public void resume() {
        renderThread.resumeExecution();
    }

    @Override
    public boolean isPaused() {
        return renderThread.isPaused();        
    }

    @Override
    public boolean isRunning() {
        return !renderThread.isPaused();
    }

    @Override
    public boolean isStopped() {
        return renderThread.isTerminated();
    }

    @Override
    public void setCamera(CCamera cameraData) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CCameraModel getCameraModel() {
        return api.getDeviceRT().getCameraModel();
    }
}
