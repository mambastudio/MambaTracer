/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.data.struct.CRay;
import cl.core.CNormalBVH;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CIntersection;
import cl.core.data.struct.CMaterial;
import cl.core.kernel.CLSource;
import cl.shapes.CMesh;
import cl.ui.mvc.viewmodel.RenderViewModel;
import coordinate.model.OrientationModel;
import coordinate.parser.OBJParser;
import coordinate.parser.attribute.MaterialT;
import coordinate.utility.Timer;
import filesystem.core.OutputFactory;
import java.net.URI;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.jocl.CL;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CResourceFactory;
import wrapper.core.CallBackFunction;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;

/**
 *
 * @author user
 */
public class RayDeviceMesh {
   
    OpenCLPlatform configuration = null; 
    
    CCamera camera = new CCamera(new CPoint3(0, 0, 9), new CPoint3(), new CVector3(0, 1, 0), 45);
    
    //image variables
    CIntBuffer imageBuffer = null;
    CIntBuffer groupBuffer = null;
    
    //group index an bound
    CIntBuffer groupIndex = null;
    CFloatBuffer groupBound = null;
    
    //viewport variables
    CStructBuffer<CCamera.CameraStruct> cameraBuffer = null;
    CIntBuffer width = null;
    CIntBuffer height = null;
    
    //Ray & intersects
    CStructBuffer<CRay> raysBuffer;
    CStructBuffer<CIntersection> isectBuffer;
    CStructBuffer<CMaterial> materialBuffer;
    
    //global count
    CIntBuffer count = null;
    
    //global and local size
    private int globalSize, localSize;
    
    //kernel    
    CKernel generateCameraRaysKernel = null;
    CKernel intersectPrimitivesKernel = null;
    CKernel fastShadeKernel = null;
    CKernel updateShadeImageKernel = null;
    CKernel groupBufferPassKernel = null;
    CKernel findBoundKernel = null;
    
    //mesh
    CFloatBuffer points = null;
    CIntBuffer faces = null;
    CIntBuffer size = null;
    
    //accelerator
    CNormalBVH bvhBuild;
    
    
    public RayDeviceMesh()
    {
        CL.setExceptionsEnabled(true);
       
        configuration = OpenCLPlatform.getDefault(CLSource.readFiles());        
        OutputFactory.print("name", configuration.device().getName());
        OutputFactory.print("type", configuration.device().getType());
        OutputFactory.print("vendor", configuration.device().getVendor());
        OutputFactory.print("speed", Long.toString(configuration.device().getSpeed()));
    }
    
    public void init(int width, int height, int globalSize, int localSize)
    {
        this.globalSize  = globalSize; this.localSize = localSize;
        
        //Init constant global variables, except mesh that is loaded after mesh is uploaded
        this.imageBuffer        = CBufferFactory.allocInt("image", configuration.context(), globalSize, READ_WRITE);
        this.groupBuffer        = CBufferFactory.allocInt("group", configuration.context(), globalSize, READ_WRITE);
        this.isectBuffer        = CBufferFactory.allocStruct("intersctions", configuration.context(), CIntersection.class, globalSize, READ_ONLY);
        this.raysBuffer         = CBufferFactory.allocStruct("rays", configuration.context(), CRay.class, globalSize, READ_WRITE);
        this.cameraBuffer       = CBufferFactory.allocStruct("camera", configuration.context(), CCamera.CameraStruct.class, 1, READ_ONLY);
        this.width              = CBufferFactory.initIntValue("width", configuration.context(), configuration.queue(), width, READ_ONLY);
        this.height             = CBufferFactory.initIntValue("height", configuration.context(), configuration.queue(), height, READ_ONLY);
        this.count              = CBufferFactory.initIntValue("count", configuration.context(), configuration.queue(), 0, READ_WRITE);
        this.groupIndex         = CBufferFactory.initIntValue("groupIndex", configuration.context(), configuration.queue(), 0, READ_ONLY);
        this.groupBound         = CBufferFactory.allocFloat("groupBound", configuration.context(), 6 , READ_WRITE);
                
        //read mesh and position camera
        initDefaultMesh();  
    }
    
    public void initMesh(String uri) {initMesh(Paths.get(uri));}
    public void initMesh(URI uri){initMesh(Paths.get(uri));}
        
    public void initMesh(Path path)
    {
        CMesh mesh = new CMesh(configuration);
        OBJParser parser = new OBJParser();
        
        //Time parsing
        Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh));
        OutputFactory.print("scene parse time", parseTime.toString());
        
        //Load scene material and group to ui
        RenderViewModel.setSceneMaterial(mesh.getMaterialList());
        
        //Time building
        Timer buildTime = Timer.timeThis(() -> mesh.buildAccelerator());
        OutputFactory.print("bvh build time", buildTime.toString());
        
        OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.reposition(camera, mesh.getBound());
        
        CResourceFactory.releaseMemory("points","faces","size","nodes","nodesSize","objects", "materials", "fastShade");
        
        this.points             = mesh.getCLPointsBuffer("points", configuration.context(), configuration.queue());
        this.faces              = mesh.getCLFacesBuffer("faces", configuration.context(), configuration.queue());
        this.size               = mesh.getCLSizeBuffer("size", configuration.context(), configuration.queue());
        this.materialBuffer     = mesh.getCLMaterialBuffer("materials", configuration.context(), configuration.queue());
        this.bvhBuild = new CNormalBVH(configuration);
        this.bvhBuild.build(mesh);           
       
        generateCameraRaysKernel = configuration.program().createKernel("generateCameraRays", cameraBuffer, raysBuffer, isectBuffer, width, height);
        intersectPrimitivesKernel = configuration.program().createKernel("intersectPrimitives", raysBuffer, isectBuffer, count, points, faces, size, bvhBuild.getCNodes(), bvhBuild.getCBounds());        
        fastShadeKernel = configuration.program().createKernel("fastShade", materialBuffer, isectBuffer);        
        updateShadeImageKernel = configuration.program().createKernel("updateShadeImage", imageBuffer, width, height, isectBuffer);        
        groupBufferPassKernel = configuration.program().createKernel("groupBufferPass", isectBuffer, groupBuffer);
        findBoundKernel = configuration.program().createKernel("findBound", groupIndex, points, faces, size, groupBound);
        
        
        //update the kernels here
    }
    
    public CCamera getCamera(){return camera;}
    
    public void execute()
    {                      
       configuration.queue().put1DRangeKernel(generateCameraRaysKernel, globalSize, localSize); 
       configuration.queue().put1DRangeKernel(intersectPrimitivesKernel, globalSize, localSize); 
       configuration.queue().put1DRangeKernel(fastShadeKernel, globalSize, localSize);
       configuration.queue().put1DRangeKernel(updateShadeImageKernel, globalSize, localSize);  
       configuration.queue().put1DRangeKernel(groupBufferPassKernel, globalSize, localSize);
       
       /*
         Why implementing this makes opencl run faster?
         Probable answer is this... https://stackoverflow.com/questions/18471170/commenting-clfinish-out-makes-program-100-faster
       */       
       configuration.queue().finish();        
        
      
    }
    
    public void readImageBuffer(CallBackFunction<IntBuffer> callback) {imageBuffer.mapReadBuffer(configuration.queue(), callback);}
    public void readGroupBuffer(CallBackFunction<IntBuffer> callback) {groupBuffer.mapReadBuffer(configuration.queue(), callback);}
   
    public void updateCamera(){this.cameraBuffer.mapWriteBuffer(configuration.queue(), cameraStruct -> 
            {
                cameraStruct[0] = camera.getCameraStruct();
                OutputFactory.print("eye", camera.position().toString());
                OutputFactory.print("dir", camera.forward().toString());
                OutputFactory.print("fov", Float.toString(camera.fov));
                
            });}
    
    public void setMaterial(int index, CMaterial material)
    {        
        this.materialBuffer.mapWriteBuffer(configuration.queue(), materialArray -> {            
            materialArray[index] = material;            
        });
    }
    
    public int getTotalSize(){return globalSize;}
    
    public CBoundingBox getBound()
    {
        return bvhBuild.getBound();
    }
    
    public CBoundingBox getGroupBound(int value)
    {
        groupBound.mapWriteBuffer(configuration.queue(), buffer -> {
           buffer.put(new float[]{Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY});
       });
        groupIndex.setArray(configuration.queue(), value);
       
       configuration.queue().put1DRangeKernel(findBoundKernel, size.get(0), 1);
       configuration.queue().finish(); // not really necessary
       
       CPoint3 min = new CPoint3();
       CPoint3 max = new CPoint3();
       
       groupBound.mapReadBuffer(configuration.queue(), buffer -> {           
           min.x = buffer.get(0); min.y = buffer.get(1); min.z = buffer.get(2);
           max.x = buffer.get(3); max.y = buffer.get(4); max.z = buffer.get(5);   
           
           //System.out.println(String.format("(%8.2f, %8.2f, %8.2f) to (%8.2f, %8.2f, %8.2f)", buffer.get(0), buffer.get(1), buffer.get(2), buffer.get(3), buffer.get(4), buffer.get(5)));
           
       });       
       return new CBoundingBox(min, max);
    }
    
    public void reposition(CBoundingBox box)
    {
        OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.repositionLocation(camera, box);
    }
    
    private void initDefaultMesh()
    {
        String cube =   "o Cube\n" +
                        "v 1.000000 -1.000000 -1.000000\n" +
                        "v 1.000000 -1.000000 1.000000\n" +
                        "v -1.000000 -1.000000 1.000000\n" +
                        "v -1.000000 -1.000000 -1.000000\n" +
                        "v 1.000000 1.000000 -0.999999\n" +
                        "v 0.999999 1.000000 1.000001\n" +
                        "v -1.000000 1.000000 1.000000\n" +
                        "v -1.000000 1.000000 -1.000000\n" +
                        "vn 0.000000 -1.000000 0.000000\n" +
                        "vn 0.000000 1.000000 0.000000\n" +
                        "vn 1.000000 0.000000 0.000000\n" +
                        "vn -0.000000 0.000000 1.000000\n" +
                        "vn -1.000000 -0.000000 -0.000000\n" +
                        "vn 0.000000 0.000000 -1.000000\n" +
                        "s off\n" +
                        "f 2//1 3//1 4//1\n" +
                        "f 8//2 7//2 6//2\n" +
                        "f 5//3 6//3 2//3\n" +
                        "f 6//4 7//4 3//4\n" +
                        "f 3//5 7//5 8//5\n" +
                        "f 1//6 4//6 8//6\n" +
                        "f 1//1 2//1 4//1\n" +
                        "f 5//2 8//2 6//2\n" +
                        "f 1//3 5//3 2//3\n" +
                        "f 2//4 6//4 3//4\n" +
                        "f 4//5 3//5 8//5\n" +
                        "f 5//6 1//6 8//6";
        
        CMesh mesh = new CMesh(configuration);
        OBJParser parser = new OBJParser();
        //parser.readString(cube, mesh);
        parser.readString(cube, mesh);
        RenderViewModel.setSceneMaterial(mesh.getMaterialList());        
        mesh.buildAccelerator();
        OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.reposition(camera, mesh.getBound());
        
        CResourceFactory.releaseMemory("points","faces","size","nodes","nodesSize","objects", "materials", "fastShadeKernel");
        
        this.points             = mesh.getCLPointsBuffer("points", configuration.context(), configuration.queue());
        this.faces              = mesh.getCLFacesBuffer("faces", configuration.context(), configuration.queue());
        this.size               = mesh.getCLSizeBuffer("size", configuration.context(), configuration.queue());
        this.materialBuffer     = mesh.getCLMaterialBuffer("materials", configuration.context(), configuration.queue());
        
        
        
        this.bvhBuild = new CNormalBVH(configuration);
        this.bvhBuild.build(mesh);
                
        generateCameraRaysKernel = configuration.program().createKernel("generateCameraRays", cameraBuffer, raysBuffer, isectBuffer, width, height);
        intersectPrimitivesKernel = configuration.program().createKernel("intersectPrimitives", raysBuffer, isectBuffer, count, points, faces, size, bvhBuild.getCNodes(), bvhBuild.getCBounds());
        fastShadeKernel = configuration.program().createKernel("fastShade", materialBuffer, isectBuffer);
        updateShadeImageKernel = configuration.program().createKernel("updateShadeImage", imageBuffer, width, height, isectBuffer);
        groupBufferPassKernel = configuration.program().createKernel("groupBufferPass", isectBuffer, groupBuffer);
        findBoundKernel = configuration.program().createKernel("findBound", groupIndex, points, faces, size, groupBound);
    }
}
