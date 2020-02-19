/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.data.CPoint2;
import coordinate.struct.StructByteArray;
import coordinate.struct.StructIntArray;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Random;
import thread.model.LambdaThread;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class SDeviceGI {
    OpenCLPlatform platform;
    BlendDisplay display;
    
    //render thread
    LambdaThread renderThread = new LambdaThread();
    
    SCameraModel cameraModel = null;
    SMesh mesh = null;
    SNormalBVH bvh = null;
    
    int width, height;
    BitmapARGB renderBitmap;
    
    //global and local size
    int globalWorkSize, localWorkSize;
    
    boolean isImageCleared = true;
    
    //CL
    private CStructTypeBuffer<SState> gStateBuffer;
    CIntBuffer gImageBuffer = null;    
    CFloatBuffer gAccumBuffer = null;
    CIntBuffer gCountBuffer = null;
    CIntBuffer gPixelIndicesBuffer = null;
    CStructTypeBuffer<SCamera> gCameraBuffer = null;    
    CStructTypeBuffer<SRay> gRaysBuffer = null;
    CStructTypeBuffer<SRay> gOcclusRaysBuffer = null;
    CStructTypeBuffer<SIsect> gIsectBuffer = null;
    CStructTypeBuffer<SPath> gBPathBuffer = null;
    SCompact compactHybrid;
    private CFloatBuffer gFrameCountBuffer;
    private CIntBuffer gTotalLights;
    private CStructTypeBuffer<SLight> gLights;
    private CStructTypeBuffer<SPath> gLightPaths;
    
    //kernels   
    private CKernel gInitCameraRaysKernel;
    private CKernel gInitIsectsKernel;
    private CKernel gInitPathsKernel;
    private CKernel gInitPixelIndicesKernel;
    private CKernel gIntersectPrimitivesKernel;
    private CKernel gUpdateBSDFIntersectKernel;
    private CKernel gLightHitPassKernel;
    private CKernel gEvaluateBSDFIntersectKernel;
    private CKernel gUpdateImageKernel;
    private CKernel gSampleBSDFRayDirectionKernel;
    private CKernel gDirectLightKernel;
    
    
    public SDeviceGI()
    {
        
    }
    
    public void init(OpenCLPlatform platform, 
                     BlendDisplay display,
                     int w, int h,
                     SMesh mesh,
                     SNormalBVH bvh,
                     SCameraModel cameraModel)
    {
        this.platform = platform;
        this.display = display;
        this.width = w;
        this.height = h;
        this.mesh = mesh;
        this.bvh = bvh;
        this.cameraModel = cameraModel;
        this.renderBitmap = new BitmapARGB(width, height);
        this.globalWorkSize = w * h;
        this.localWorkSize = 250;

        initBuffers();
        initKernels();
        compactHybrid.init(gIsectBuffer, gPixelIndicesBuffer, gCountBuffer);
    }
    
    public void initBuffers()
    {
        gStateBuffer         = CBufferFactory.allocStructType("gState", platform.context(), SState.class, 1, READ_WRITE);
        gPixelIndicesBuffer  = CBufferFactory.allocInt("gPixelIndices", platform.context(), globalWorkSize, READ_WRITE);
        gImageBuffer         = CBufferFactory.allocInt("gImage", platform.context(), globalWorkSize, READ_WRITE);
        gAccumBuffer         = CBufferFactory.allocFloat("gAccum", platform.context(), globalWorkSize * 4, READ_WRITE);
        gCountBuffer         = CBufferFactory.initIntValue("gCount", platform.context(), platform.queue(), globalWorkSize, READ_WRITE);
        gRaysBuffer          = CBufferFactory.allocStructType("gRays", platform.context(), SRay.class, globalWorkSize, READ_WRITE);
        gOcclusRaysBuffer    = CBufferFactory.allocStructType("gOcclusRays", platform.context(), SRay.class, globalWorkSize, READ_WRITE);   
        gCameraBuffer        = CBufferFactory.allocStructType("gCamera", platform.context(), SCamera.class, 1, READ_WRITE);
        gBPathBuffer         = CBufferFactory.allocStructType("gPaths", platform.context(), SPath.class, globalWorkSize, READ_WRITE);
        gIsectBuffer         = CBufferFactory.allocStructType("gIsects", platform.context(), SIsect.class, globalWorkSize, READ_WRITE);
        gFrameCountBuffer    = CBufferFactory.initFloatValue("gFrameCount", platform.context(), platform.queue(), 1, READ_WRITE);
        compactHybrid        = new SCompact(platform);       
        gTotalLights         = CBufferFactory.initIntValue("totalLights", platform.context(), platform.queue(), 0, READ_WRITE);
        gLights              = CBufferFactory.allocStructType("lights", platform.context(), SLight.class, 1, READ_WRITE);
        gLightPaths          = CBufferFactory.allocStructType("lightPaths", platform.context(), SPath.class, globalWorkSize, READ_WRITE);
    }
    
    public void initKernels()
    {
        gInitCameraRaysKernel           = platform.createKernel("InitCameraRayData", gCameraBuffer, gRaysBuffer);
        gInitPathsKernel                = platform.createKernel("InitPathData", gBPathBuffer);
        gInitIsectsKernel               = platform.createKernel("InitIntersection", gIsectBuffer);
        gInitPixelIndicesKernel         = platform.createKernel("InitIntDataToIndex", gPixelIndicesBuffer);
        gIntersectPrimitivesKernel      = platform.createKernel("IntersectPrimitives", gRaysBuffer, gIsectBuffer, gCountBuffer, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds());
        //gUpdateBSDFIntersectKernel      = platform.createKernel("UpdateBSDFIntersect", gIsectBuffer, gRaysBuffer, gBPathBuffer, gPixelIndicesBuffer, gCountBuffer);
        gLightHitPassKernel             = platform.createKernel("LightHitPass", gIsectBuffer, gRaysBuffer, gBPathBuffer, mesh.clMaterials(), gAccumBuffer, gPixelIndicesBuffer, gCountBuffer);
      //  gEvaluateBSDFIntersectKernel    = platform.createKernel("EvaluateBSDFIntersect", gIsectBuffer, gBPathBuffer, mesh.clMaterials(), gPixelIndicesBuffer, gCountBuffer);
        gUpdateImageKernel              = platform.createKernel("UpdateImage", gAccumBuffer, gFrameCountBuffer, gImageBuffer);
        gSampleBSDFRayDirectionKernel   = platform.createKernel("SampleBSDFRayDirection", gIsectBuffer, gRaysBuffer, gBPathBuffer, mesh.clMaterials(), gPixelIndicesBuffer, gStateBuffer, gCountBuffer);
        gDirectLightKernel              = platform.createKernel("DirectLight", gBPathBuffer, gIsectBuffer, gLights, gTotalLights, gOcclusRaysBuffer, gAccumBuffer, gPixelIndicesBuffer, gCountBuffer, gStateBuffer, mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds());
    }
    
    public void start()
    {
        //set screen black first       
        this.renderBitmap.reset(true);
        this.display.imageFill("RENDER", renderBitmap);
        this.isImageCleared = false;
        this.initLight();
        this.initFrameCount();
        this.updateCamera();
        renderThread.startExecution(()-> {
            //execute pause             
            loop();            
            
            platform.queue().finish();
            incrementFrameCount();
            //renderThread.pauseExecution(); 
        });      
    }
    
    private void loop()
    {
        renderThread.chill();
        
        
        //reset intersection count
        
        
        platform.executeKernel1D(gInitCameraRaysKernel, globalWorkSize, localWorkSize);
        platform.executeKernel1D(gInitPathsKernel, globalWorkSize, localWorkSize); 
        platform.executeKernel1D(gInitIsectsKernel, globalWorkSize, localWorkSize);  
        platform.executeKernel1D(gInitPixelIndicesKernel, globalWorkSize, localWorkSize);
        resetCount();
        for(int pathLength = 1; pathLength<=1; pathLength++)
        {   
            platform.executeKernel1D(gIntersectPrimitivesKernel, globalWorkSize, localWorkSize);           
            platform.executeKernel1D(gLightHitPassKernel, globalWorkSize, localWorkSize);         
            compactHybrid.execute();           
            directLightEvaluation();
            allocateState();//set new seed state            
            platform.executeKernel1D(gSampleBSDFRayDirectionKernel, globalWorkSize, localWorkSize);
           
        }
        updateImage();
        
        renderThread.chill();
        
        //renderThread.pauseExecution();
    }
    
    public void directLightEvaluation()
    {
        allocateState();//set new seed state        
        platform.executeKernel1D(gDirectLightKernel, globalWorkSize, localWorkSize);
    }
    
    public void clearImage()
    {
        this.renderBitmap.reset(false);
        this.display.imageFill("RENDER", renderBitmap);
        this.isImageCleared = true;
    }
    
    public boolean isImageCleared()
    {
        return this.isImageCleared;
    }
    
    public void initLight()
    {
        StructByteArray<SLight> lights = new StructByteArray<>(SLight.class);
        StructIntArray<SFace> faces = new StructIntArray<>(SFace.class, mesh.getCount());        
        faces.setIntArray(mesh.getTriangleFacesArray());      
        int lightCount = 0;
        
        for(int i = 0; i<faces.size(); i++)
        {
            SFace face = faces.get(i);
            SMaterial material = mesh.clMaterials().get(face.getMaterialIndex());
            if(material.emitterEnabled) 
            {
                lights.add(new SLight(i));
                lightCount++;
            }            
        }
        
        CResourceFactory.releaseMemory("lights");
        gTotalLights.mapWriteValue(platform.queue(), lightCount);
        gLights  = CBufferFactory.allocStructType("lights", platform.context(), lights, READ_WRITE);
        gLights.mapWriteBuffer(platform.queue(), buffer->{});//write into device (FIXME)          
        gDirectLightKernel.resetPutArgs(gBPathBuffer, gIsectBuffer, gLights, gTotalLights, gOcclusRaysBuffer, gAccumBuffer, gPixelIndicesBuffer, gCountBuffer, gStateBuffer, 
                mesh.clMaterials(), mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds()); 
       
        System.out.println("light count : " +lightCount);
    }
    
    public void updateCamera(){
        gCameraBuffer.mapWriteBuffer(platform.queue(), cameraStruct -> 
        {
            SCamera cam = cameraModel.getCameraStruct();
            cam.setDimension(new CPoint2(width, height));                
            cameraStruct.set(cam, 0);
        });
    }
    
    private void resetCount()
    {
        gCountBuffer.mapWriteValue(platform.queue(), globalWorkSize);
    }
    
    private void incrementFrameCount()
    {
        gFrameCountBuffer.mapWriteValue(platform.queue(), gFrameCountBuffer.mapReadValue(platform.queue()) + 1);
    }
    
    private void initFrameCount()
    {
        gFrameCountBuffer.mapWriteValue(platform.queue(), 1);
    }
    
    private void updateImage()
    {
        platform.executeKernel1D(gUpdateImageKernel, globalWorkSize, localWorkSize);
        gImageBuffer.mapReadBuffer(platform.queue(), buffer->{            
            renderBitmap.writeColor(buffer.array(), 0, 0, width, height);
            this.display.imageFill("RENDER", renderBitmap);
        });
    }
    
    private void allocateState()
    {
        //init seed, image dimension and increment frame count
        gStateBuffer.mapWriteBuffer(platform.queue(), states->{
            SState state = states.get(0); //we use one state
            
            //seed for current frame count
            int seed0 = BigInteger.probablePrime(30, new Random()).intValue();
            int seed1 = BigInteger.probablePrime(30, new Random()).intValue();
            
            state.setSeed(seed0, seed1);                       
            state.setFrameCount(gFrameCountBuffer.mapReadValue(platform.queue()));
        });
    }
}
