/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.renderer;

import bitmap.display.StaticDisplay;
import bitmap.image.BitmapARGB;
import cl.core.CBoundingBox;
import cl.core.CCamera;
import cl.core.device.RayDeviceBoxSVM;
import cl.core.CRay;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.device.RayDeviceBox;
import cl.core.device.RayDeviceMesh;
import coordinate.model.OrientationModel;
import coordinate.utility.Timer;
import java.nio.file.Path;
import thread.model.KernelThread;

/**
 *
 * @author user
 */
public class SimpleRender extends KernelThread{
    RayDeviceMesh device; 
    
    int globalSize = 700;
    int localSize = 100;
    
    BitmapARGB bitmap;
    int[] bufferImage;
    StaticDisplay display;
    
    
    OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
    
    @Override
    public void execute() {
        Timer timer = new Timer();
        timer.start();
        
        device.updateCamera();
        device.execute();
        device.readImageBuffer(buffer-> {
            buffer.get(bufferImage);
            bitmap.writeColor(bufferImage, 0, 0, globalSize, globalSize);
            display.imageFill(bitmap);
        });
        
        timer.end();
        
        pauseKernel();
        
        chill();
       // System.out.printf("%.2f rays/second \n", device.getTotalSize()/timer.seconds());       
    }
    
    public boolean init() 
    {
        device = new RayDeviceMesh();
        device.init(globalSize, localSize);
        bufferImage = new int[globalSize * globalSize];
        bitmap = new BitmapARGB(globalSize, globalSize);
        
        return true;
    }
    
    public void launch(StaticDisplay display) {
        this.display = display;     
        display.translationDepth.addListener((observable, old_value, new_value) -> {
            orientation.translateDistance(device.getCamera(), new_value.floatValue());     
            resumeKernel();
        });
        
        display.translationXY.addListener((observable, old_value, new_value) -> {
            orientation.rotateX(device.getCamera(), (float) new_value.getX());
            orientation.rotateY(device.getCamera(), (float) new_value.getY());
            resumeKernel();
        });
        
        init();
               
        startKernel();
    }
    
    public boolean close() {
        return true;
    }
    
    
    public void initMesh(Path path)
    {
        device.initMesh(path);
    }
}
