/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import bitmap.display.BlendDisplay;
import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CCamera.CameraStruct;
import cl.core.CCompact;
import cl.core.CImageFrame;
import cl.core.CNormalBVH;
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
   
    //image, image count and seed
    private CStructTypeBuffer<CState> gState;
    private CImageFrame  gFrameImage;   
       
    //pixel indices and camera
    private CIntBuffer   gPixelIndices;   
    private CStructTypeBuffer<CameraStruct> gCamera;
        
    //kernels   
    private CKernel rInitCameraRaysKernel;
    private CKernel rInitIsectsKernel;
    private CKernel rInitPathsKernel;
    private CKernel gInitPixelIndicesKernel;    
    private CKernel gIntersectTestKernel;
    private CKernel gInitOcclusionHitsKernel;
    private CKernel gSampleLightKernel;
    private CKernel gGenerateShadowRaysKernel;
    private CKernel gOcclusionTestKernel;
    private CKernel gEvaluateBsdfExplicitKernel;
    private CKernel gProcessIntersection;
    private CKernel gLightHitPassKernel;  
    private CKernel rEvaluateBSDFIntersectKernel;
    private CKernel rSampleBSDFRayDirectionKernel;
    
    
    //Prefix sum
    CCompact compact;
    
    private CIntBuffer   gCount;   
   
    //Ray & intersects
    private CStructTypeBuffer<CRay> gRays;   
    private CStructTypeBuffer<CIntersection> gIsects;    
    private CStructTypeBuffer<CPath> gPaths;
    
    //lights, light path, occlusion hits and shadow rays
    private CStructTypeBuffer<CLight> gLights;
    private CIntBuffer gTotalLights;
    private CStructTypeBuffer<CPath> gLightPaths;
    private CStructTypeBuffer<CRay> gShadowRays;
    private CIntBuffer gOcclusionHits;
    
    //mesh and accelerator
    CMesh mesh;
    CNormalBVH bvhBuild; 
    
    //global and local size
    private int globalSize, localSize;
    
    //render thread
    LambdaThread renderThread = new LambdaThread();
    Random random = new Random();
    
    public RayDeviceMeshRender()
    {
        
    }    
        
    public void initBuffers()
    {
        gFrameImage.initBuffers();
    }   
    
    @Override
    public void setAPI(TracerAPI api) {
        this.api = api;
        this.globalSize = api.getGlobalSizeForDevice(RENDER);
        this.localSize = 1;
        
        this.gPixelIndices          = api.allocInt("rPixels", globalSize, READ_WRITE);
        this.gIsects                = api.allocStructType("renderIsects", CIntersection.class, globalSize, READ_WRITE);
        this.gPaths                 = api.allocStructType("rPaths", CPath.class, globalSize, READ_WRITE);
        this.gRays                  = api.allocStructType("renderRays", CRay.class, globalSize, READ_WRITE);
        this.gShadowRays            = api.allocStructType("gShadowRays", CRay.class, globalSize, READ_WRITE);
        this.gCount                 = api.intValue("renderCount", globalSize, READ_WRITE);
        this.gCamera                = api.allocStructType("camera", CameraStruct.class, 1, READ_WRITE);
        this.gOcclusionHits         = api.allocInt("gOcclusionHits", globalSize, READ_WRITE);
         
        this.gState                 = api.allocStructType("gState", CState.class, 1, READ_WRITE);
       
        this.gFrameImage            = new CImageFrame(api.configurationCL(), api.getImageWidth(RENDER_IMAGE), api.getImageHeight(RENDER_IMAGE));
      
        //scene light buffer
        this.gLightPaths            = api.allocStructType("lightPaths", CPath.class, globalSize, READ_WRITE);
        
        //this are dynamic hence dummy initials
        this.gLights                = api.allocStructType("lights", CLight.class, 1, READ_WRITE);
        
        //currently none        
        this.gTotalLights           = api.intValue("totalLights", 0, READ_WRITE);
        
        
        
        //raytracing compaction
        compact = new CCompact(api);
        compact.initPrefixSumFrom(gIsects, gCount);
        compact.initIsects(gIsects);
        compact.initPixels(gPixelIndices);
        
        
    }

    @Override
    public void set(CMesh mesh, CNormalBVH bvhBuild) {
        this.mesh = mesh;
        this.bvhBuild = bvhBuild;
        
        this.rInitCameraRaysKernel          = api.createKernel("InitCameraRayData", gCamera, gRays);
        this.rInitPathsKernel               = api.createKernel("InitPathData", gPaths);
        this.rInitIsectsKernel              = api.createKernel("initIntersection", gIsects);
        this.gInitPixelIndicesKernel        = api.createKernel("initPixelIndices", gPixelIndices);
        this.gInitOcclusionHitsKernel       = api.createKernel("InitIntData", gOcclusionHits); 
        this.gSampleLightKernel             = api.sampleLightsKernel("sampleLight", gLightPaths, gLights, gTotalLights, gCount, gState, mesh);
        this.gGenerateShadowRaysKernel      = api.createKernel("GenerateShadowRays", gLightPaths, gIsects, gRays, gCount);
        this.gIntersectTestKernel           = api.createIntersectionKernel("intersectPrimitives", gRays, gIsects, gCount, mesh, bvhBuild);
        this.gOcclusionTestKernel           = api.createOcclusionKernel("intersectOcclusion", gShadowRays, gOcclusionHits, gCount, mesh, bvhBuild);
        this.gEvaluateBsdfExplicitKernel    = api.createKernel("EvaluateBsdfExplicit", gIsects, gOcclusionHits, gPaths, mesh.clMaterials(), gPixelIndices, gCount);
        this.gProcessIntersection           = api.createKernel("UpdateBSDFIntersect", gIsects, gRays, gPaths, gPixelIndices, gCount);
        this.gLightHitPassKernel            = api.createKernel("LightHitPass", gIsects, gPaths, mesh.clMaterials(), gFrameImage.getFrameAccum(), gCamera, gCount);
        this.rEvaluateBSDFIntersectKernel   = api.createKernel("EvaluateBSDFIntersect", gIsects, gPaths, mesh.clMaterials(), gCamera, gCount);
        this.rSampleBSDFRayDirectionKernel  = api.createKernel("SampleBSDFRayDirection", gIsects, gRays, gPaths, gPixelIndices, gState, gCount);
        
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
        
        CResourceFactory.releaseMemory("lights");
        gTotalLights.mapWriteValue(api.configurationCL().queue(), lightCount);
        gLights  = api.allocStructType("lights", lights, lightCount, READ_WRITE);
        
        gSampleLightKernel.resetPutArgs(gLightPaths, gLights, gTotalLights, gCount, gState, 
                mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize()); 
        
        System.out.println("light count : " +lightCount);
    }

    @Override
    public void execute() {       
        updateCamera();        
        initBuffers();
        initLight();
        renderThread.startExecution(()->{                 
            //path trace here            
            loop();
            //add frame count
            gFrameImage.incrementFrameCount();
        });
        
    }
    
    public void loop()
    {         
        
        //pause level
        renderThread.chill();   
        
        //init intersections
        api.configurationCL().queue().put1DRangeKernel(rInitIsectsKernel, globalSize, localSize);
        //init path
        api.configurationCL().queue().put1DRangeKernel(rInitPathsKernel, globalSize, localSize);
        //init camera rays
        api.configurationCL().queue().put1DRangeKernel(rInitCameraRaysKernel, globalSize, localSize);
        //init pixel indices
        api.execute1D(gInitPixelIndicesKernel, globalSize, localSize);
        //reset intersection count
        gCount.mapWriteValue(api.configurationCL().queue(), globalSize);
        
        
        //init seed, image dimension and increment frame count
        gState.mapWriteBuffer(api.configurationCL().queue(), states->{
            CState state = states.get(0); //we use one state
            
            //seed for current frame count
            int seed0 = BigInteger.probablePrime(30, new Random()).intValue();
            int seed1 = BigInteger.probablePrime(30, new Random()).intValue();
            
            state.setSeed(seed0, seed1);
            state.setDimension(
                    api.getImageWidth(RENDER_IMAGE), 
                    api.getImageHeight(RENDER_IMAGE));            
            state.incrementFrameCount();            
        });
        
        //path trace
        for(int i = 0; i<2; i++)
        {
            //rCount.mapReadBuffer(api.configurationCL().queue(), buffer -> System.out.println(buffer.get())); 
            //intersect primitives
            api.configurationCL().queue().put1DRangeKernel(gIntersectTestKernel, globalSize, localSize);
            
            //transfer data to path, from intersection (such as bsdf, hitpoint)
            api.configurationCL().queue().put1DRangeKernel(gProcessIntersection, globalSize, localSize);
                       
            //implicit light hit
            api.configurationCL().queue().put1DRangeKernel(gLightHitPassKernel, globalSize, localSize);
                  
            //compact intersection and pixels
            compact.executePrefixSum();
            compact.compactIsects(globalSize);
            compact.compactPixels(globalSize);
            
            //evaluate bsdf
            api.configurationCL().queue().put1DRangeKernel(this.rEvaluateBSDFIntersectKernel, globalSize, localSize);
            
            //direct light evaluation
            directLightEvaluation();
            
            //sample new directions
            api.configurationCL().queue().put1DRangeKernel(this.rSampleBSDFRayDirectionKernel, globalSize, localSize);
        }
        
        //process image (tonemapping, average luminance, etc)
        gFrameImage.processImage();
       
        //ensure queue clears up (makes it faster)
        api.configurationCL().queue().finish();
        
         //read image
        api.readImageFromDevice(RENDER, RENDER_IMAGE);   
        
        
    }
    
    /* 
     * init hit buffer
     * sample light
     * generate shadow rays
     * intersect shadow rays
     * evaluate direct light bsdf
    */
    public void directLightEvaluation()
    {
        api.execute1D(gInitOcclusionHitsKernel, globalSize, localSize);
        api.execute1D(gSampleLightKernel, globalSize, localSize);
        api.execute1D(gGenerateShadowRaysKernel, globalSize, localSize);
        api.execute1D(gOcclusionTestKernel, globalSize, localSize);
        api.execute1D(gEvaluateBsdfExplicitKernel, globalSize, localSize);
    }

    @Override
    public void updateCamera() {
        
        CCamera rtCamera = api.getDevice(RAYTRACE).getCamera();
        this.gCamera.mapWriteBuffer(api.configurationCL().queue(), cameraStruct -> 
            {
                CCamera.CameraStruct cam = rtCamera.getCameraStruct();
                cam.setDimension(new CPoint2(api.getImageSize(RAYTRACE_IMAGE)));
                cameraStruct.set(cam, 0);
                
                OutputFactory.print("eye", rtCamera.position().toString());
                OutputFactory.print("dir", rtCamera.forward().toString());
                OutputFactory.print("fov", Float.toString(rtCamera.fov));
                
            });
    }

    @Override
    public CCamera getCamera() {
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
            gFrameImage.getFrameARGB().mapReadBuffer(api.configurationCL().queue(), callback);
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
    
}
