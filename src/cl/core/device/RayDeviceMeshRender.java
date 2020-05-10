/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.CAccelerator;
import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CCameraModel;
import cl.core.CCompact;
import cl.core.CImageFrame;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.core.api.MambaAPIInterface.DeviceType.RENDER;
import static cl.core.api.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.core.api.RayDeviceInterface;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.RENDER_BUFFER;
import cl.core.data.CPoint2;
import cl.core.data.struct.CFace;
import cl.core.data.struct.CPath;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CLight;
import cl.core.data.struct.CMaterial;
import cl.core.data.struct.CRay;
import cl.core.data.struct.CState;
import cl.main.TracerAPI;
import cl.shapes.CMesh;
import coordinate.parser.attribute.MaterialT;
import coordinate.struct.StructByteArray;
import coordinate.struct.StructIntArray;
import filesystem.core.OutputFactory;
import java.math.BigInteger;
import java.nio.IntBuffer;
import java.util.Random;
import thread.model.LambdaThread;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.CallBackFunction;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class RayDeviceMeshRender implements RayDeviceInterface<TracerAPI, IntBuffer, BlendDisplay, MaterialT> {
    //API
    TracerAPI api;
   
    BlendDisplay display;
    
    //render thread
    LambdaThread renderThread = new LambdaThread();
    
    CCameraModel cameraModel = null;
    CMesh mesh = null;
    CAccelerator bvh = null;
    
    int width, height;
    BitmapARGB renderBitmap;
    
    //global and local size
    int globalSize, localSize;
    
    boolean isImageCleared = true;
    
    //Frame
    CImageFrame frame = null;
    
    //CL
    CStructTypeBuffer<CState> gStateBuffer;
    CIntBuffer gCounterBuffer = null;
    CIntBuffer gPixelIndicesBuffer = null;
    CStructTypeBuffer<CCamera> gCameraBuffer = null;    
    CStructTypeBuffer<CRay> gRaysBuffer = null;
    CStructTypeBuffer<CRay> gOcclusRaysBuffer = null;
    CStructTypeBuffer<CIntersection> gIsectBuffer = null;
    CStructTypeBuffer<CPath> gBPathBuffer = null;
    CCompact gCompact;
    private CIntBuffer gStartNode = null;
    private CIntBuffer gTotalLights;
    private CStructTypeBuffer<CLight> gLights;
  
    //kernels   
    private CKernel gInitCameraRaysJitterKernel;
    private CKernel gInitIsectsKernel;
    private CKernel gInitPathsKernel;
    private CKernel gInitPixelIndicesKernel;
    private CKernel gIntersectPrimitivesKernel;  
    private CKernel gLightHitPassKernel;    
    private CKernel gSampleBSDFRayDirectionKernel;
    private CKernel gDirectLightKernel;
    
    public RayDeviceMeshRender()
    {
        
    }    
    
    public void initBuffers()
    {
        frame.initBuffers();
    }   
    
    @Override
    public void setAPI(TracerAPI api) {
        this.api = api;
        this.globalSize = api.getGlobalSizeForDevice(RENDER);
        this.localSize = 100;
        this.frame = new CImageFrame(api.configurationCL(), api.getImageWidth(RENDER_IMAGE), api.getImageHeight(RENDER_IMAGE));
        gStartNode           = api.allocIntValue("gStartNode", 0, READ_WRITE);
        gStateBuffer         = api.allocStructType("gState", CState.class, 1, READ_WRITE);
        gPixelIndicesBuffer  = api.allocInt("gPixelIndices",  globalSize, READ_WRITE);        
        gCounterBuffer       = api.allocIntValue("gCount", globalSize, READ_WRITE);
        gRaysBuffer          = api.allocStructType("gRays", CRay.class, globalSize, READ_WRITE);
        gOcclusRaysBuffer    = api.allocStructType("gOcclusRays", CRay.class, globalSize, READ_WRITE);   
        gCameraBuffer        = api.allocStructType("gCamera", CCamera.class, 1, READ_WRITE);
        gBPathBuffer         = api.allocStructType("gPaths", CPath.class, globalSize, READ_WRITE);
        gIsectBuffer         = api.allocStructType("gIsects", CIntersection.class, globalSize, READ_WRITE);
        gTotalLights         = api.allocIntValue("totalLights",  0, READ_WRITE);
        gLights              = api.allocStructType("lights", CLight.class, 1, READ_WRITE);
        gCompact             = new CCompact(api.configurationCL());       
        gCompact.init(gIsectBuffer, gPixelIndicesBuffer, gCounterBuffer);
    }

    @Override
    public void set(CMesh mesh, CAccelerator bvhBuild) {
        this.mesh = mesh;
        this.bvh = bvhBuild;
        
        gStartNode.mapWriteValue(api.configurationCL().queue(), bvhBuild.getStartNodeIndex());
        
        gInitCameraRaysJitterKernel     = api.createKernel("InitCameraRayDataJitter", gCameraBuffer, gRaysBuffer, gStateBuffer);
        gInitPathsKernel                = api.createKernel("InitPathData", gBPathBuffer);
        gInitIsectsKernel               = api.createKernel("InitIntersection", gIsectBuffer);
        gInitPixelIndicesKernel         = api.createKernel("InitIntDataToIndex", gPixelIndicesBuffer);
        gIntersectPrimitivesKernel      = api.createKernel("IntersectPrimitives", gRaysBuffer, gIsectBuffer, gCounterBuffer, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds(), gStartNode);
        gLightHitPassKernel             = api.createKernel("LightHitPass", gIsectBuffer, gRaysBuffer, gBPathBuffer,  gTotalLights, mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), frame.getFrameAccum(), gPixelIndicesBuffer, gCounterBuffer);
        gSampleBSDFRayDirectionKernel   = api.createKernel("SampleBSDFRayDirection", gIsectBuffer, gRaysBuffer, gBPathBuffer, mesh.clMaterials(), gPixelIndicesBuffer, gStateBuffer, gCounterBuffer);
        gDirectLightKernel              = api.createKernel("DirectLight", gBPathBuffer, gIsectBuffer, gLights, gTotalLights, gOcclusRaysBuffer, frame.getFrameAccum(), gPixelIndicesBuffer, gCounterBuffer, gStateBuffer, mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds(), gStartNode);
    
    }

    @Override
    public void setMaterial(int index, MaterialT aterial) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public void initLight()
    {    
        StructByteArray<CLight> lights = new StructByteArray<>(CLight.class);
        StructIntArray<CFace> faces = new StructIntArray<>(CFace.class, mesh.getCount());        
        faces.setIntArray(mesh.getTriangleFacesArray());      
        int lightCount = 0;
                        
        for(int i = 0; i<faces.size(); i++)
        {
            CFace face = faces.get(i);            
            CMaterial material = mesh.clMaterials().get(face.getMaterialIndex());
            
            if(material.emitterEnabled) 
            {
                lights.add(new CLight(i));
                lightCount++;
            }            
        }
        System.out.println("light count : " +lightCount);
        CResourceFactory.releaseMemory("lights");
        gTotalLights.mapWriteValue(api.configurationCL().queue(), lightCount);
        gLights  = CBufferFactory.allocStructType("lights", api.configurationCL().context(), lights, READ_WRITE);
        gLights.mapWriteBuffer(api.configurationCL().queue(), buffer->{});//write into device (FIXME)          
        gDirectLightKernel.resetPutArgs(gBPathBuffer, gIsectBuffer, gLights, gTotalLights, gOcclusRaysBuffer, frame.getFrameAccum(), gPixelIndicesBuffer, gCounterBuffer, gStateBuffer, 
                mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds()); 
    }

    @Override
    public void execute() {  
        updateCamera();
        initBuffers();
        initLight();
        renderThread.startExecution(()-> {
            //execute pause             
            loop();
            frame.incrementFrameCount();
           // System.out.println("kubafu");
        });   
        
    }
    
    public void loop()
    {         
        renderThread.chill();
         
        allocateState();//set new seed state
        api.execute1D(gInitCameraRaysJitterKernel, globalSize, localSize);
        api.execute1D(gInitPathsKernel, globalSize, localSize); 
        api.execute1D(gInitIsectsKernel, globalSize, localSize);  
        api.execute1D(gInitPixelIndicesKernel, globalSize, localSize);
        resetCounter();
        for(int pathLength = 1; pathLength<=2; pathLength++)
        {               
            api.execute1D(gIntersectPrimitivesKernel, globalSize, localSize);           
            api.execute1D(gLightHitPassKernel, globalSize, localSize);       
            gCompact.execute();           
            directLightEvaluation();
            allocateState();//set new seed state            
            api.execute1D(gSampleBSDFRayDirectionKernel, globalSize, localSize);
           
        }
        //process image (tonemapping, average luminance, etc)
        frame.processImage();
              
        //ensure queue clears up (makes it faster)
        api.configurationCL().queue().finish();
        
        //read image
        api.readImageFromDevice(RENDER, RENDER_IMAGE);  
    }
    private void resetCounter()
    {
        gCounterBuffer.mapWriteValue(api.configurationCL().queue(), globalSize);
    }
    
    public void directLightEvaluation()
    {
        allocateState();//set new seed state        
        api.execute1D(gDirectLightKernel, globalSize, localSize);
    }
    
    private void allocateState()
    {
        //init seed, image dimension and increment frame count
        gStateBuffer.mapWriteBuffer(api.configurationCL().queue(), states->{
            CState state = states.get(0); //we use one state
            
            //seed for current frame count
            int seed0 = BigInteger.probablePrime(30, new Random()).intValue();
            int seed1 = BigInteger.probablePrime(30, new Random()).intValue();
            
            state.setSeed(seed0, seed1);   
            state.setFrameCount(frame.getFrameCount().mapReadValue(api.configurationCL().queue()));
        });
    }

    @Override
    public void updateCamera() {
        CCameraModel rtCamera = api.getDevice(RAYTRACE).getCameraModel();
        this.gCameraBuffer.mapWriteBuffer(api.configurationCL().queue(), cameraStruct -> 
            {
                CCamera cam = rtCamera.getCameraStruct();
                cam.setDimension(new CPoint2(api.getImageSize(RAYTRACE_IMAGE)));
                cameraStruct.set(cam, 0);
                
                OutputFactory.print("eye", rtCamera.position().toString());
                OutputFactory.print("dir", rtCamera.forward().toString());
                OutputFactory.print("fov", Float.toString(rtCamera.fov));                
            });        
    }

    @Override
    public CCameraModel getCameraModel() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getTotalSize() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBoundingBox getBound() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setLocalSize(int local) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
    public boolean isPaused() {
        return renderThread.isPaused();       
    }

    @Override
    public boolean isRunning() {
        return !renderThread.isPaused();
      
    }

    @Override
    public void setCamera(CCamera camera) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void resume() {
       renderThread.resumeExecution();
    }

    @Override
    public void setGlobalSize(int globalSize) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void readBuffer(DeviceBuffer name, CallBackFunction callback) {
        if(name == RENDER_BUFFER)
            frame.getFrameARGB().mapReadBuffer(api.configurationCL().queue(), callback);
    }

    @Override
    public CBoundingBox getGroupBound(int value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean isStopped() {
        return renderThread.isTerminated();        
    }

    @Override
    public void setShadeType(ShadeType type) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public ShadeType getShadeType() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CBoundingBox getPriorityBound() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setPriorityBound(CBoundingBox bound) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
