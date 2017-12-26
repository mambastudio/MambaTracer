/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.device;

import cl.core.CBVHAccelerator.CBVHNode;
import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.CRay;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.shapes.CMesh;
import coordinate.model.OrientationModel;
import coordinate.parser.OBJParser;
import coordinate.utility.Timer;
import filesystem.core.OutputFactory;
import java.net.URI;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.WRITE_ONLY;
import wrapper.core.CResourceFactory;
import wrapper.core.CallBackFunction;
import wrapper.core.PlatformConfiguration;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;
import wrapper.core.buffer.CStructBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class RayDeviceMesh {
    String directory = "raytracing/";
   
    PlatformConfiguration configuration = null; 
    
    CCamera camera = new CCamera(new CPoint3(0, 0, 9), new CPoint3(), new CVector3(0, 1, 0), 45);
    
    //kernel variables
    CIntBuffer imageBuffer = null;
    CStructBuffer<CCamera.CameraStruct> cameraBuffer = null;
    CIntBuffer width = null;
    CIntBuffer height = null;
    
    //global and local size
    private int globalSize, localSize;
    
    //kernel
    CKernel raytracingKernel = null;
    
    //mesh
    CFloatBuffer points = null;
    CIntBuffer faces = null;
    CIntBuffer size = null;
    
    //accelerator
    CStructBuffer<CBVHNode> nodes = null;
    CIntBuffer nodesSize = null;
    CIntBuffer objects = null;
    
    
    public RayDeviceMesh()
    {
        String source1 = CLFileReader.readFile(directory, "Common.cl");
        String source2 = CLFileReader.readFile(directory, "Primitive.cl");
        String source3 = CLFileReader.readFile(directory, "SimpleTrace.cl");
        
        configuration = PlatformConfiguration.getDefault(source1, source2, source3);        
        OutputFactory.print("name", configuration.device().getName());
        OutputFactory.print("type", configuration.device().getType());
        OutputFactory.print("vendor", configuration.device().getVendor());
        OutputFactory.print("speed", Long.toString(configuration.device().getSpeed()));
    }
    
    public void init(int globalSize, int localSize)
    {
        this.globalSize  = globalSize; this.localSize = localSize;
        
        //Init constant global variables
        this.imageBuffer        = CBufferFactory.allocInt("image", configuration.context(), globalSize * globalSize, WRITE_ONLY);
        this.cameraBuffer       = CBufferFactory.allocStruct("camera", configuration.context(), CCamera.CameraStruct.class, 1, READ_ONLY);
        this.width              = CBufferFactory.initIntValue("width", configuration.context(), configuration.queue(), globalSize, READ_ONLY);
        this.height             = CBufferFactory.initIntValue("height", configuration.context(), configuration.queue(), globalSize, READ_ONLY);
        
        //read mesh and position camera
        initDefaultMesh();  
    }
    
    public void initMesh(String uri) {initMesh(Paths.get(uri));}
    public void initMesh(URI uri){initMesh(Paths.get(uri));}
        
    public void initMesh(Path path)
    {
        CMesh mesh = new CMesh();
        OBJParser parser = new OBJParser();
        
        //Time parsing
        Timer parseTime = Timer.timeThis(() -> parser.read(path.toString(), mesh));
        OutputFactory.print("scene parse time", parseTime.toString());
        
        //Time building
        Timer buildTime = Timer.timeThis(() -> mesh.buildAccelerator());
        OutputFactory.print("bvh build time", buildTime.toString());
        
        OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.reposition(camera, mesh.getBound());
        
        CResourceFactory.releaseMemory("points","faces","size","nodes","nodesSize","objects");
        
        this.points             = mesh.getCLPointsBuffer("points", configuration.context(), configuration.queue());
        this.faces              = mesh.getCLFacesBuffer("faces", configuration.context(), configuration.queue());
        this.size               = mesh.getCLSizeBuffer("size", configuration.context(), configuration.queue());
        this.nodes              = mesh.getCLBVHNodeArray("nodes", configuration.context(), configuration.queue());
        this.nodesSize          = mesh.getCLBVHNodeArraySize("nodesSize", configuration.context(), configuration.queue());
        this.objects            = mesh.getCLObjectIdBuffer("objects", configuration.context(), configuration.queue());
        
        raytracingKernel = configuration.program().createKernel("traceMesh", imageBuffer, cameraBuffer, width, height, points, faces, size, nodes, nodesSize, objects);
    }
    
    public CCamera getCamera(){return camera;}
    
    public void execute(){configuration.queue().put1DRangeKernel(raytracingKernel, globalSize * globalSize, localSize);}
    
    public void readImageBuffer(CallBackFunction<IntBuffer> callback) {imageBuffer.mapReadBuffer(configuration.queue(), callback);}
    public void updateCamera(){this.cameraBuffer.mapWriteBuffer(configuration.queue(), cameraStruct -> 
            {
                cameraStruct[0] = camera.getCameraStruct();
                OutputFactory.print("eye", camera.position().toString());
                OutputFactory.print("dir", camera.forward().toString());
                OutputFactory.print("fov", Float.toString(camera.fov));
                
            });}
    public int getTotalSize(){return globalSize * globalSize;}
    
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
        
        CMesh mesh = new CMesh();
        OBJParser parser = new OBJParser();
        parser.readString(cube, mesh);
        mesh.buildAccelerator();
        OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
        orientation.reposition(camera, mesh.getBound());
        
        CResourceFactory.releaseMemory("points","faces","size","nodes","nodesSize","objects");
        
        this.points             = mesh.getCLPointsBuffer("points", configuration.context(), configuration.queue());
        this.faces              = mesh.getCLFacesBuffer("faces", configuration.context(), configuration.queue());
        this.size               = mesh.getCLSizeBuffer("size", configuration.context(), configuration.queue());
        this.nodes              = mesh.getCLBVHNodeArray("nodes", configuration.context(), configuration.queue());
        this.nodesSize          = mesh.getCLBVHNodeArraySize("nodesSize", configuration.context(), configuration.queue());
        this.objects            = mesh.getCLObjectIdBuffer("objects", configuration.context(), configuration.queue());
        
        raytracingKernel = configuration.program().createKernel("traceMesh", imageBuffer, cameraBuffer, width, height, points, faces, size, nodes, nodesSize, objects);
    }
}
