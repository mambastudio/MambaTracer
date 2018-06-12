package cl.core.device;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import cl.core.CCamera;
import cl.core.CCamera.CameraStruct;
import cl.shapes.CBox;
import java.nio.IntBuffer;
import wrapper.core.CBufferFactory;
import wrapper.core.CKernel;
import static wrapper.core.CMemory.READ_ONLY;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.CallBackFunction;
import wrapper.core.OpenCLPlatform;
import wrapper.core.svm.CSVMIntBuffer;
import wrapper.core.svm.CSVMStructBuffer;
import wrapper.util.CLFileReader;

/**
 *
 * @author user
 */
public class RayDeviceBoxSVM {
    String directory = "C:\\Users\\user\\Documents\\Java\\jocl\\cl\\raytracing/";
   
    OpenCLPlatform configuration = null; 
    
    //kernel variables
    CSVMIntBuffer imageBuffer = null;
    CSVMStructBuffer<CameraStruct> camera = null;
    CSVMIntBuffer width = null;
    CSVMIntBuffer height = null;
    CSVMStructBuffer<CBox> boxes = null;
    
    //global and local size
    private int globalSize, localSize;
    
    //kernel
    CKernel kernelTraceBox = null;
   
      
    public RayDeviceBoxSVM()
    {
        String source1 = CLFileReader.readFile(directory, "Common.cl");
        String source2 = CLFileReader.readFile(directory, "Primitive.cl");
        String source3 = CLFileReader.readFile(directory, "SimpleTrace.cl");
        
        configuration = OpenCLPlatform.getDefault(source1, source2, source3);
    }
    
    public void init(int globalSize, int localSize)
    {
        this.globalSize  = globalSize; this.localSize = localSize;
        
        this.imageBuffer = CBufferFactory.allocSVMInt(configuration.context(), globalSize * globalSize, READ_WRITE);
        this.camera      = CBufferFactory.allocSVMStruct(configuration.context(), CameraStruct.class, 1, READ_WRITE);
        this.width       = CBufferFactory.allocSVMInt(configuration.context(), 1, READ_ONLY);
        this.height      = CBufferFactory.allocSVMInt(configuration.context(), 1, READ_ONLY);
        this.boxes       = CBufferFactory.allocSVMStruct(configuration.context(), CBox.class, 1, READ_ONLY);
        
        kernelTraceBox = configuration.program().createSVMKernel("traceBox", imageBuffer, camera, width, height, boxes);
        
        width.mapWriteBuffer(configuration.queue(), buffer -> buffer.put(0, globalSize));
        height.mapWriteBuffer(configuration.queue(), buffer -> buffer.put(0, globalSize));
        boxes.mapWriteBuffer(configuration.queue(), array -> array[0] = new CBox());
    }
    
   
    
    public void execute()
    {           
        configuration.queue().put1DRangeKernel(kernelTraceBox, globalSize * globalSize, localSize);
    }
    
    public void mapReadImageBuffer(CallBackFunction<IntBuffer> callback) 
    {
        imageBuffer.mapReadBuffer(configuration.queue(), callback);
    }
    
    public void setCamera(CCamera camera)
    {
        this.camera.mapWriteBuffer(configuration.queue(), cameraStruct -> cameraStruct[0] = camera.getCameraStruct()); 
    }
    
    public int getTotalSize(){
        return globalSize * globalSize;
    }
}
