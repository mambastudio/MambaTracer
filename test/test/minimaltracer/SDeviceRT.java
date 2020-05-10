/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.model.OrientationModel;
import coordinate.parser.obj.OBJParser;
import thread.model.LambdaThread;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructTypeBuffer;

/**
 *
 * @author user
 */
public class SDeviceRT {
    OpenCLPlatform platform;
    BlendDisplay display;
    
    //render thread
    LambdaThread raytraceThread = new LambdaThread();
    
    SCameraModel cameraModel = new SCameraModel(new CPoint3(0, 0, -9), new CPoint3(), new CVector3(0, 1, 0), 45);
    SMesh mesh = null;
    SNormalBVH bvh = null;
    
    final int width, height;
    BitmapARGB raytraceBitmap;
    
    //global and local size
    int globalWorkSize, localWorkSize;
    
    //CL
    CIntBuffer imageBuffer = null;      
    CStructTypeBuffer<SCamera> cameraBuffer = null;    
    CStructTypeBuffer<SRay> raysBuffer = null;
    CStructTypeBuffer<SIsect> isectBuffer = null;
    CIntBuffer count = null;
    
    CKernel initCameraRaysKernel = null;
    CKernel intersectPrimitivesKernel = null;
    CKernel fastShadeKernel = null;
    CKernel backgroundShadeKernel = null;
       
    public SDeviceRT(int w, int h)
    {
        this.width = w; 
        this.height = h;
        this.raytraceBitmap = new BitmapARGB(w, h);
        this.globalWorkSize = width * height;
        this.localWorkSize = 1;
    }
    
    public void init(OpenCLPlatform platform, BlendDisplay display)
    {
        this.platform = platform;
        this.display = display;
        initBuffers();
        initDefaultMesh();
        initKernels();        
        
    }
    
    public void initBuffers()
    {
        raysBuffer          = CBufferFactory.allocStructType("rays", platform.context(), SRay.class, globalWorkSize, READ_WRITE);
        cameraBuffer        = CBufferFactory.allocStructType("camera", platform.context(), SCamera.class, 1, READ_WRITE);
        count               = CBufferFactory.initIntValue("count", platform.context(), platform.queue(), globalWorkSize, READ_WRITE);
        isectBuffer         = CBufferFactory.allocStructType("intersections", platform.context(), SIsect.class, globalWorkSize, READ_WRITE);
        imageBuffer         = CBufferFactory.allocInt("image", platform.context(), globalWorkSize, READ_WRITE);        
    }
    
    public void initKernels()
    {      
        initCameraRaysKernel                = platform.createKernel("InitCameraRayData", cameraBuffer, raysBuffer);
        intersectPrimitivesKernel           = platform.createKernel("IntersectPrimitives", raysBuffer, isectBuffer, count, mesh.clPoints(), mesh.clNormals(), mesh.clFaces(), mesh.clSize(), bvh.getCNodes(), bvh.getCBounds());
        fastShadeKernel                     = platform.createKernel("fastShade", mesh.clMaterials(), isectBuffer, imageBuffer);
        backgroundShadeKernel               = platform.createKernel("backgroundShade", isectBuffer, cameraBuffer, imageBuffer);
        
    }
    
    public void updateCamera(){
        cameraBuffer.mapWriteBuffer(platform.queue(), cameraStruct -> 
        {
            SCamera cam = cameraModel.getCameraStruct();
            cam.setDimension(new CPoint2(getWidth(), getHeight()));                
            cameraStruct.set(cam, 0);
        });
    }
  
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
        
        platform.executeKernel1D(initCameraRaysKernel, globalWorkSize, localWorkSize);
        platform.executeKernel1D(intersectPrimitivesKernel, globalWorkSize, localWorkSize);
        platform.executeKernel1D(backgroundShadeKernel, globalWorkSize, localWorkSize);          
        platform.executeKernel1D(fastShadeKernel, globalWorkSize, localWorkSize);
        
        platform.queue().finish();
        
        readImageFromDevice();
        
        raytraceThread.chill();
    }
    
    public void readImageFromDevice() {      
        imageBuffer.mapReadBuffer(platform.queue(), buffer-> {
            this.raytraceBitmap.writeColor(buffer.array(), 0, 0, width, height);
            this.display.imageFill("RAYTRACE", raytraceBitmap);
            
        }); 
                    
    }
    
    public int getWidth()
    {
        return width;
    }
    
    public int getHeight()
    {
        return height;
    }
    
    public void pause() {
        raytraceThread.pauseExecution();
    }

    public void stop() {
        raytraceThread.stopExecution();
    }
   
    public boolean isPaused() {
        return raytraceThread.isPaused();
    }

    public boolean isRunning() {
        return !raytraceThread.isPaused();
    }

    public void resume() {
        raytraceThread.resumeExecution();
    }
    
    public boolean isStopped() {
        return raytraceThread.isTerminated();
    }
    
    public void reposition(SBound box)
    {
        OrientationModel<CPoint3, CVector3, SRay, SBound> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.repositionLocation(cameraModel, box);     
    }
    
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
        mesh = new SMesh(platform);   
        OBJParser parser = new OBJParser();        
        parser.readString(cube, mesh);
        mesh.initCLBuffers();
        
        //modify materials
        mesh.clMaterials().mapWriteBuffer(platform.queue(), materials -> {
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
            SMaterial emitter = materials.get(6);
            //emitter.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki            
            emitter.setEmitter(1, 1, 1);
            emitter.setEmitterEnabled(true);
                       
            SMaterial right = materials.get(3);           
            right.setDiffuse(0, 0.8f, 0);
            
            SMaterial left = materials.get(7); 
            //left.setEmitter(1, 1, 1);
            //left.setEmitterEnabled(true);
            left.setDiffuse(0.8f, 0f, 0);
            
            SMaterial back = materials.get(2);           
            back.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
            
            SMaterial ceiling = materials.get(1);           
            ceiling.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
            
            SMaterial floor = materials.get(0);           
            floor.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
            
            SMaterial smallbox = materials.get(4);  
            smallbox.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki
            //smallbox.setEmitter(1, 1, 1);
            //smallbox.setEmitterEnabled(true);
            
            SMaterial tallbox = materials.get(5);           
            tallbox.setDiffuse(0.7647f, 0.6902f, 0.5686f);  //khaki            
        });
                
        //build accelerator
        bvh = new SNormalBVH(platform);
        bvh.build(mesh);      
        
        //Set cameraModel new position
        reposition(mesh.getBound());
        updateCamera();
    }
    
    public SMesh getMesh()
    {
        return mesh;
    }
    
    public SNormalBVH getBVH()
    {
        return bvh;
    }
    
    public SCameraModel getCameraModel()
    {
        return cameraModel;
    }
}
