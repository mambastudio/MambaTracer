/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.minimaltracer;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import coordinate.model.OrientationModel;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import org.jocl.CL;
import test.minimaltracer.cl.CSource;
import wrapper.core.OpenCLPlatform;

/**
 * FXML Controller class
 *
 * @author user
 */
public class SUserInterfaceFXMLController implements Initializable {
    @FXML
    StackPane pane;
    @FXML
    Button button1;
    @FXML
    Button button2;
    /**
     * Initializes the controller class.
     */
    
    private BlendDisplay display = null;    
    private OpenCLPlatform platform = null;
    
    //devices
    private SDeviceRT deviceRT = null;
    private SDeviceGI deviceGI = null;
    
    private final OrientationModel<CPoint3, CVector3, SRay, SBound> orientation = new OrientationModel(CPoint3.class, CVector3.class);
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        //device and display
        this.deviceRT = new SDeviceRT(500, 500);
        this.deviceGI = new SDeviceGI();
        
        this.display = new BlendDisplay("RAYTRACE", "RENDER");        
        
        display.translationDepth.addListener((observable, old_value, new_value) -> {   
            if(deviceGI.isImageCleared())
            {
                orientation.translateDistance(deviceRT.cameraModel, new_value.floatValue() * deviceRT.mesh.getBound().getMaximumExtent());     
                deviceRT.resume();
            }
        });
        
        display.translationXY.addListener((observable, old_value, new_value) -> {    
            if(deviceGI.isImageCleared())
            {
                orientation.rotateX(deviceRT.cameraModel, (float) new_value.getX());
                orientation.rotateY(deviceRT.cameraModel, (float) new_value.getY());
                deviceRT.resume();
            }
        });
            
        //add display component
        pane.getChildren().add(display);
        
        //render, pause, stop and back to rt 
        button2.disableProperty().set(true);
        
        button1.setOnAction(e -> {
            switch (button1.getText()) {
                case "Render":
                    button1.disableProperty().set(true);
                    button2.disableProperty().set(false);
                    button2.setText("Pause");
                    deviceGI.start();
                    break;                
                default:
                    break;
            }
        });
        
        button2.setOnAction(e -> {
            switch (button2.getText()) {
                case "Pause":
                    button1.disableProperty().set(false);
                    button2.setText("Stop");
                    break;
                case "Stop":
                    button1.disableProperty().set(false);
                    button2.setText("Edit");
                    break;
                case "Edit":
                    button1.setText("Render");
                    button2.disableProperty().set(true);
                    deviceGI.clearImage();
                    break;
                default:
                    break;
            }
        });
        
         
        //Init device
        initRT();
    }    
    
    
    public void initRT()
    {
        CL.setExceptionsEnabled(true);
        platform = OpenCLPlatform.getDefault(CSource.readFiles());
        
        //set black image
        display.set("RAYTRACE", new BitmapARGB(deviceRT.getWidth(), deviceRT.getHeight(), true));   
        
        deviceRT.init(platform, display);
        deviceRT.start();
        deviceGI.init(platform, display, 
                      deviceRT.getWidth(), deviceRT.getHeight(), 
                      deviceRT.getMesh(), deviceRT.getBVH(), deviceRT.getCameraModel());
    }
}
