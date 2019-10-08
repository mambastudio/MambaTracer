/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CCompaction;
import cl.core.CNormalBVH;
import cl.core.api.MambaAPIInterface;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.core.api.MambaAPIInterface.DeviceType.RENDER;
import static cl.core.api.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.core.api.RayDeviceInterface;
import static cl.core.api.RayDeviceInterface.DeviceBuffer.RENDER_BUFFER;
import cl.core.data.struct.CPath;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CRay;
import cl.shapes.CMesh;
import coordinate.parser.attribute.MaterialT;
import filesystem.core.OutputFactory;
import java.math.BigInteger;
import java.util.Random;
import thread.model.LambdaThread;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CallBackFunction;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class RayDeviceMeshRender implements RayDeviceInterface {
    //API
    MambaAPIInterface api;
    
    //frame buffer
    private CIntBuffer   rFrame;
    private CFloatBuffer rFrameCount;
    private CIntBuffer   rWidth;
    private CIntBuffer   rHeight;
    private CFloatBuffer rAccum;
    private CFloatBuffer rTotalLogLum;
    
    //count and seed
    private CIntBuffer   rCount; 
    
    private CIntBuffer   random0;
    private CIntBuffer   random1;
    
    //camera buffer
    private CStructBuffer<CCamera.CameraStruct> rCamera;
        
    //kernels
    private CKernel rInitFrameKernel;
    private CKernel rInitAccumKernel;    
    private CKernel rInitCameraRaysKernel;
    private CKernel rInitIsectsKernel;
    private CKernel rInitPathsKernel;
    private CKernel rIntersectPrimitivesKernel;
    private CKernel rUpdateBSDFIntersectKernel;
    private CKernel rLightHitPassKernel;  
    private CKernel rEvaluateBSDFIntersectKernel;
    private CKernel rSampleBSDFRayDirectionKernel;
    private CKernel rTotalLogLuminanceKernel;
    private CKernel rUpdateFrameImageKernel;
    
    //Compaction
    CCompaction compactIsect;
    
     //Ray & intersects
    CStructTypeBuffer<CRay> rRays;
    CStructTypeBuffer<CIntersection> rIsects;
    CStructTypeBuffer<CPath> rPaths;
    
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
        //init frame buffer
        api.configurationCL().queue().put1DRangeKernel(rInitFrameKernel, globalSize, localSize);        
        //init accumulation buffer
        api.configurationCL().queue().put1DRangeKernel(rInitAccumKernel, globalSize, localSize);        
        //init frame count
        rFrameCount.mapWriteValue(api.configurationCL().queue(), 1);      
        
    }   
    
    @Override
    public void setAPI(MambaAPIInterface api) {
        this.api = api;
        this.globalSize = api.getGlobalSizeForDevice(RENDER);
        this.localSize = 1;
        
        this.rIsects                = CBufferFactory.allocStructType("renderIsects", api.configurationCL().context(), CIntersection.class, globalSize, READ_WRITE);
        this.rPaths                 = CBufferFactory.allocStructType("rPaths", api.configurationCL().context(), CPath.class, globalSize, READ_WRITE);
        this.rRays                  = CBufferFactory.allocStructType("renderRays", api.configurationCL().context(), CRay.class, globalSize, READ_WRITE);
        this.rCount                 = CBufferFactory.initIntValue("renderCount", api.configurationCL().context(), api.configurationCL().queue(), globalSize, READ_WRITE);
        this.rCamera                = CBufferFactory.allocStruct("camera", api.configurationCL().context(), CCamera.CameraStruct.class, 1, READ_WRITE);
        this.rWidth                 = CBufferFactory.initIntValue("width", api.configurationCL().context(), api.configurationCL().queue(), api.getImageSize(RENDER_IMAGE).x, READ_ONLY);
        this.rHeight                = CBufferFactory.initIntValue("height", api.configurationCL().context(), api.configurationCL().queue(), api.getImageSize(RENDER_IMAGE).y, READ_ONLY);
        
        this.random0                = CBufferFactory.allocInt("random0", api.configurationCL().context(), 1, READ_WRITE);
        this.random1                = CBufferFactory.allocInt("random1", api.configurationCL().context(), 1, READ_WRITE);
        
        this.rFrame                 = CBufferFactory.allocInt("frameBuffer", api.configurationCL().context(), globalSize, READ_WRITE);
        this.rFrameCount            = CBufferFactory.allocFloat("frameCount", api.configurationCL().context(), 1, READ_WRITE);
        this.rAccum                 = CBufferFactory.allocFloat("accumBuffer", api.configurationCL().context(), globalSize*4, READ_WRITE); //float * 4 = float4
        this.rTotalLogLum           = CBufferFactory.allocFloat("totalLogLum", api.configurationCL().context(), 1, READ_WRITE);
        
        compactIsect = new CCompaction(api.configurationCL());
        compactIsect.init(rIsects, rCount);
    }

    @Override
    public void set(CMesh mesh, CNormalBVH bvhBuild) {
        this.mesh = mesh;
        this.bvhBuild = bvhBuild;
        
        this.rInitAccumKernel               = api.configurationCL().program().createKernel("initFloat4DataXYZ", rAccum);
        this.rInitFrameKernel               = api.configurationCL().program().createKernel("initIntDataRGB", rFrame);
        this.rInitCameraRaysKernel          = api.configurationCL().program().createKernel("InitCameraRayData", rCamera, rRays, rWidth, rHeight);
        this.rInitPathsKernel               = api.configurationCL().program().createKernel("InitPathData", rPaths);
        this.rInitIsectsKernel              = api.configurationCL().program().createKernel("InitIsectData", rIsects);
        this.rIntersectPrimitivesKernel     = api.configurationCL().program().createKernel("intersectPrimitives", rRays, rIsects, rCount, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvhBuild.getCNodes(), bvhBuild.getCBounds());
        this.rUpdateBSDFIntersectKernel     = api.configurationCL().program().createKernel("UpdateBSDFIntersect", rIsects, rRays, rPaths, rWidth, rHeight, rCount);
        this.rLightHitPassKernel            = api.configurationCL().program().createKernel("LightHitPass", rIsects, rPaths, mesh.clMaterials(), rAccum, rWidth, rHeight, rCount);
        this.rEvaluateBSDFIntersectKernel   = api.configurationCL().program().createKernel("EvaluateBSDFIntersect", rIsects, rPaths, mesh.clMaterials(), rWidth, rHeight, rCount);
        this.rSampleBSDFRayDirectionKernel  = api.configurationCL().program().createKernel("SampleBSDFRayDirection", rIsects, rRays, rPaths, rWidth, rHeight, rCount, random0, random1, rFrameCount);
        this.rTotalLogLuminanceKernel       = api.configurationCL().program().createKernel("TotalLogLuminance", rAccum, rFrameCount, rTotalLogLum);
        this.rUpdateFrameImageKernel        = api.configurationCL().program().createKernel("UpdateFrameImage", rAccum, rFrame, rFrameCount);
    }

    @Override
    public void setMaterial(int index, MaterialT aterial) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void execute() {       
        updateCamera();        
        initBuffers();   
        renderThread.startExecution(()->{                 
            //path trace here
            loop();
            //add frame count
            rFrameCount.mapWriteValue(api.configurationCL().queue(), rFrameCount.mapReadValue(api.configurationCL().queue()) + 1);
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
        //reset intersection count
        rCount.mapWriteValue(api.configurationCL().queue(), globalSize);
        
        random0.mapWriteValue(api.configurationCL().queue(), BigInteger.probablePrime(30, new Random()).intValue()); 
        random1.mapWriteValue(api.configurationCL().queue(), BigInteger.probablePrime(30, new Random()).intValue());
        
        //path trace
        for(int i = 0; i<2; i++)
        {
            //rCount.mapReadBuffer(api.configurationCL().queue(), buffer -> System.out.println(buffer.get())); 
            //intersect primitives
            api.configurationCL().queue().put1DRangeKernel(rIntersectPrimitivesKernel, globalSize, localSize);
            
            //deal with bsdf
            api.configurationCL().queue().put1DRangeKernel(rUpdateBSDFIntersectKernel, globalSize, localSize);
                       
            //implicit light hit
            api.configurationCL().queue().put1DRangeKernel(this.rLightHitPassKernel, globalSize, localSize);
                  
            //compact intersection
            compactIsect.execute(); 
            
            //evaluate bsdf
            api.configurationCL().queue().put1DRangeKernel(this.rEvaluateBSDFIntersectKernel, globalSize, localSize);
            
            //sample new directions
            api.configurationCL().queue().put1DRangeKernel(this.rSampleBSDFRayDirectionKernel, globalSize, localSize);
        }
        
        //api.configurationCL().queue().put1DRangeKernel(this.rTotalLogLuminanceKernel, globalSize, localSize);
        api.configurationCL().queue().put1DRangeKernel(this.rUpdateFrameImageKernel, globalSize, localSize);
        
        api.configurationCL().queue().finish();
        
         //read image
        api.readImageFromDevice(RENDER, RENDER_IMAGE);   
        
        
    }

    @Override
    public void updateCamera() {
        
        CCamera rtCamera = api.getDevice(RAYTRACE).getCamera();
        this.rCamera.mapWriteBuffer(api.configurationCL().queue(), cameraStruct -> 
            {
                cameraStruct[0] = rtCamera.getCameraStruct();
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
            rFrame.mapReadBuffer(api.configurationCL().queue(), callback);
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
