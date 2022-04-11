/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package raytrace;

import cl.ui.fx.BlendDisplay;
import bitmap.image.BitmapARGB;
import static cl.abstracts.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import static cl.abstracts.MambaAPIInterface.getGlobal;
import cl.abstracts.RayDeviceInterface;
import cl.algorithms.CEnvironment;
import cl.algorithms.CTextureApplyPass;
import cl.data.CPoint2;
import cl.data.CPoint3;
import cl.data.CVector3;
import cl.scene.CMesh;
import cl.scene.CNormalBVH;
import cl.struct.CBound;
import cl.struct.CBsdf;
import cl.struct.CCamera;
import cl.struct.CCameraModel;
import cl.struct.CIntersection;
import cl.struct.CMaterial2;
import cl.struct.CRay;
import cl.struct.CTextureData;
import cl.ui.fx.Overlay;
import coordinate.model.OrientationModel;
import javafx.application.Platform;
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
public class RaytraceDevice implements RayDeviceInterface<
        RaytraceAPI, 
        BlendDisplay, 
        CMaterial2, 
        CMesh,
        CNormalBVH,
        CBound,
        CCameraModel, 
        CCamera>{
    
    OpenCLConfiguration configuration;
    BlendDisplay display;
    
    //API
    RaytraceAPI api;
    
    //render thread
    LambdaThread raytraceThread = new LambdaThread();
    
    CCameraModel cameraModel = new CCameraModel(new CPoint3(0, 0, -9), new CPoint3(), new CVector3(0, 1, 0), 45);
    CMesh mesh = null;
    CNormalBVH bvh = null;
    
    private final int width;
    private final int height;
    private final BitmapARGB raytraceBitmap;
    private final Overlay overlay;
    
    //global and local size
    int globalWorkSize, localWorkSize;
    
    //priority bound, that is the main focus for the ray trace (e.g. selected object)
    CBound priorityBound;
    
    //CL
    CMemory<IntValue> imageBuffer = null;      
    CMemory<CCamera> cameraBuffer = null;    
    CMemory<CRay> raysBuffer = null;
    CMemory<CIntersection> isectBuffer = null;
    CMemory<IntValue> count = null;
    CMemory<IntValue> groupBuffer = null;
    CMemory<CTextureData> texBuffer = null;
    CMemory<CBsdf> bsdfBuffer = null;
    
    CKernel initCameraRaysKernel = null;
    CKernel intersectPrimitivesKernel = null;
    CKernel fastShadeKernel = null;
    CKernel backgroundShadeKernel = null;
    CKernel updateGroupbufferShadeImageKernel = null;
    CKernel textureInitPassKernel = null;
    CKernel updateToTextureColorRTKernel = null;
    CKernel setupBSDFRaytraceKernel = null;
    
    CTextureApplyPass texApplyPass = null;
    
    //environment map if any
    RaytraceEnvironment envmap = null;
    
    //on screen intersection mouse click
    CMemory<IntValue> groupIndex = null;  
    CMemory<FloatValue> groupBound = null;
    CKernel findBoundKernel = null;
    
    public RaytraceDevice(int w, int h)
    {
        this.width = w; 
        this.height = h;
        this.raytraceBitmap = new BitmapARGB(w, h);
        this.overlay = new Overlay(w, h);
        this.globalWorkSize = width * height;
        this.localWorkSize = 250;
    }

    @Override
    public void setAPI(RaytraceAPI api) {
        this.api = api;
        init(api.getConfigurationCL(), api.getDisplay(BlendDisplay.class));
    }
    
    public void initBuffers()
    {
        raysBuffer          = configuration.createBufferB(CRay.class, globalWorkSize, READ_WRITE);
        cameraBuffer        = configuration.createBufferB(CCamera.class, 1, READ_WRITE);
        count               = configuration.createFromI(IntValue.class, new int[]{globalWorkSize}, READ_WRITE);
        isectBuffer         = configuration.createBufferB(CIntersection.class, globalWorkSize, READ_WRITE);
        imageBuffer         = configuration.createBufferI(IntValue.class, globalWorkSize, READ_WRITE);        
        groupBuffer         = configuration.createBufferI(IntValue.class, globalWorkSize, READ_WRITE);
        texBuffer           = configuration.createBufferI(CTextureData.class, globalWorkSize, READ_WRITE);
        bsdfBuffer          = configuration.createBufferB(CBsdf.class, globalWorkSize, READ_WRITE);
        texApplyPass        = new CTextureApplyPass(api, texBuffer, count);
        
        groupIndex          = configuration.createBufferI(IntValue.class, 1, READ_WRITE);
        groupBound          = configuration.createBufferF(FloatValue.class, 6, READ_WRITE);
    }
    
    public void initKernels()
    {        
        initCameraRaysKernel                = configuration.createKernel("InitCameraRayData", cameraBuffer, raysBuffer);
        intersectPrimitivesKernel           = configuration.createKernel("IntersectPrimitives", raysBuffer, isectBuffer, count, mesh.clPoints(), mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getNodes(), bvh.getBounds());
        fastShadeKernel                     = configuration.createKernel("fastShade", isectBuffer, bsdfBuffer, imageBuffer);
        backgroundShadeKernel               = configuration.createKernel("backgroundShade", isectBuffer, cameraBuffer, imageBuffer, raysBuffer, envmap.getRgbCL(), envmap.getEnvMapSize());
        updateGroupbufferShadeImageKernel   = api.getConfigurationCL().createKernel("updateGroupbufferShadeImage", isectBuffer, cameraBuffer, groupBuffer);
        textureInitPassKernel               = configuration.createKernel("textureInitPassRT", bsdfBuffer, isectBuffer, texBuffer);
        setupBSDFRaytraceKernel             = configuration.createKernel("SetupBSDFRaytrace", isectBuffer, raysBuffer, bsdfBuffer, mesh.clMaterials());
        updateToTextureColorRTKernel        = configuration.createKernel("updateToTextureColorRT", bsdfBuffer, texBuffer);
        findBoundKernel                     = configuration.createKernel("findBound", groupIndex, mesh.clPoints(),  mesh.clTexCoords(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), groupBound);
    }
    
    public void findBound(int instance, CBound bound)
    {
        groupBound.mapWriteMemory(buffer->{
            buffer.set(0, new FloatValue(Float.POSITIVE_INFINITY));
            buffer.set(1, new FloatValue(Float.POSITIVE_INFINITY));
            buffer.set(2, new FloatValue(Float.POSITIVE_INFINITY));
            buffer.set(3, new FloatValue(Float.NEGATIVE_INFINITY));
            buffer.set(4, new FloatValue(Float.NEGATIVE_INFINITY));
            buffer.set(5, new FloatValue(Float.NEGATIVE_INFINITY));
        });
        groupIndex.setCL(new IntValue(instance));
        
        int local = 128;
        int global = getGlobal(mesh.clSize().getCL().v, local);
        
        configuration.execute1DKernel(findBoundKernel, global, local);
        
        //return bound and mark changes
        groupBound.mapReadMemory((buffer)->{
            bound.minimum.x = buffer.get(0).v;
            bound.minimum.y = buffer.get(1).v;
            bound.minimum.z = buffer.get(2).v;
            bound.maximum.x = buffer.get(3).v;
            bound.maximum.y = buffer.get(4).v;
            bound.maximum.z = buffer.get(5).v;
        });
        
    }
    
    public boolean isCoordinateAnInstance(double x, double y)
    {
        return overlay.isInstance(x, y);
    }
    
    public int getInstanceValue(double x, double y)
    {
        return overlay.get(x, y);
    }
    
    public Overlay getOverlay()
    {
        return overlay;
    }
    
    public int getWidth()
    {
        return width;
    }
    
    public int getHeight()
    {
        return height;
    }
    
    @Override
    public void outputImage() {      
        //transfer data from opencl to cpu
        imageBuffer.transferFromDevice();
        groupBuffer.transferFromDevice();
        //write to bitmap and overlay
        raytraceBitmap.writeColor((int[]) imageBuffer.getBufferArray(), 0, 0, width, height);
        overlay.copyToArray((int[])groupBuffer.getBufferArray());
        //image fill
        Platform.runLater(()-> display.imageFill(RAYTRACE_IMAGE.name(), raytraceBitmap));
        
    }

    @Override
    public void set(CMesh mesh, CNormalBVH bvhBuild) {
        this.mesh = mesh;
        this.bvh = bvhBuild;        
        this.priorityBound = bvhBuild.getBound();
    }

    @Override
    public void setGlobalSize(int globalSize) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setLocalSize(int localSize) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void execute() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void pause() {
        raytraceThread.pauseExecution();
    }

    @Override
    public void stop() {
        raytraceThread.stopExecution();
    }

    @Override
    public void resume() {
        raytraceThread.resumeExecution();
    }

    @Override
    public boolean isPaused() {
        return raytraceThread.isPaused();
    }

    @Override
    public boolean isRunning() {
        return !raytraceThread.isPaused();
    }

    @Override
    public boolean isStopped() {
        return raytraceThread.isTerminated();
    }

    @Override
    public void updateCamera() {
        CCamera cam = cameraModel.getCameraStruct();
        cam.setDimension(new CPoint2(getWidth(), getHeight()));
        cameraBuffer.setCL(cam);
    }

    @Override
    public void setCamera(CCamera cameraData) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public CBound getBound(){
        return bvh.getBound();
    } 
    
    @Override
    public CBound getPriorityBound()
    {
        return priorityBound;
    }

    @Override
    public CCameraModel getCameraModel() {
        return cameraModel;
    }
    
    private void init(OpenCLConfiguration platform, BlendDisplay display)
    {
        this.configuration = platform;
        this.display = display;
        this.envmap = api.getEnvironmentalMapCL();
        initBuffers();
        initKernels();        
    }
    
    public void reposition(CBound box)
    {
        OrientationModel<CPoint3, CVector3, CRay, CBound> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.repositionLocation(cameraModel, box);     
    }    
    
    public CMesh getMesh()
    {
        return mesh;
    }
    
    public CNormalBVH getBVH()
    {
        return bvh;
    }
    
    @Override
    public void start()
    {     
       raytraceThread.startExecution(()-> {
            //execute pause             
            loop();
            raytraceThread.pauseExecution();       
        });      
    }
    
    private void loop()
    {        
        if(cameraModel.isSynched(cameraBuffer.get(0)))
            raytraceThread.chill();       
        updateCamera();
        configuration.execute1DKernel(initCameraRaysKernel, globalWorkSize, localWorkSize);
        configuration.execute1DKernel(intersectPrimitivesKernel, globalWorkSize, localWorkSize);
        configuration.execute1DKernel(setupBSDFRaytraceKernel, globalWorkSize, localWorkSize);
        
        //pass texture
        configuration.execute1DKernel(textureInitPassKernel, globalWorkSize, localWorkSize);
        texApplyPass.process();
        configuration.execute1DKernel(updateToTextureColorRTKernel, globalWorkSize, localWorkSize);
        
        configuration.execute1DKernel(backgroundShadeKernel, globalWorkSize, localWorkSize); 
        configuration.execute1DKernel(fastShadeKernel, globalWorkSize, localWorkSize);
        configuration.execute1DKernel(updateGroupbufferShadeImageKernel, globalWorkSize, localWorkSize);
        outputImage();
        configuration.finish();
      
        
        
        //raytraceThread.chill();
    }
    
}
