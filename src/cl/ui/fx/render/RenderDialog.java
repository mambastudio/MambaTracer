/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.render;

import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import java.io.IOException;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import jfx.dialog.DialogAbstract;
import cl.ui.fx.main.TracerAPI;
import javafx.scene.layout.BorderPane;

/**
 *
 * @author user
 */
public class RenderDialog extends DialogAbstract{
    @FXML
    Button resumeBtn;
    @FXML
    Button pauseBtn;
    @FXML
    Button stopBtn;
    @FXML
    Button editBtn;
    @FXML
    Button renderBtn;
    
    @FXML
    StackPane renderPane;
    
    private final TracerAPI api;
    
    public RenderDialog(TracerAPI api)
    {
        BorderPane box = initFXMLComponent();
        
        this.api = api;
        
        this.renderPane.getChildren().add(api.getBlendDisplayGI());
        
        this.setContent(box);
        this.setSupplier((buttonType)-> null);
        this.removeBorder();
        
    }
    
    private BorderPane initFXMLComponent()
    {
        BorderPane box = new BorderPane();
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource(
            "RenderDialogFXML.fxml"));
        fxmlLoader.setRoot(box);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        } 
        
        resumeBtn.setDisable(true);
        editBtn.setDisable(true);
        renderBtn.setDisable(true);
        
        pauseBtn.setOnAction(e->{
            pauseBtn.setDisable(true);
            resumeBtn.setDisable(false);
            api.getDeviceGI().pause();
        });
        
        resumeBtn.setOnAction(e->{
            pauseBtn.setDisable(false);
            resumeBtn.setDisable(true);
            api.getDeviceGI().resume();
        });
        
        stopBtn.setOnAction(e->{
            pauseBtn.setDisable(true);
            resumeBtn.setDisable(true);
            editBtn.setDisable(false);
            renderBtn.setDisable(false);
            api.getDeviceGI().stop();
        });
        
        renderBtn.setOnAction(e->{
            pauseBtn.setDisable(false);
            resumeBtn.setDisable(true);
            editBtn.setDisable(true);
            renderBtn.setDisable(true);
            api.getDeviceGI().start();
        });
        
        editBtn.setOnAction(e->{
            resume();
            api.setDevicePriority(RAYTRACE);
        });
        
        return box;
    }
    
}
