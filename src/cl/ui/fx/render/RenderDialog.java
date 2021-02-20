/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.render;

import cl.abstracts.MambaAPIInterface;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import java.io.IOException;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import jfx.dialog.DialogAbstract;
import cl.ui.fx.main.TracerAPI;
import javafx.beans.binding.Bindings;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.StringProperty;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.util.converter.NumberStringConverter;
import static jfx.util.BindingFX.bindBidirectionalStringAndDouble;
import static jfx.util.ImplUtils.convertToTextFieldDouble;

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
    
    @FXML
    Slider gammaSlider;
    @FXML
    TextField gammaTextField;
    @FXML
    Slider exposureSlider;
    @FXML
    TextField exposureTextField;
    
    DoubleProperty gammaProperty = new SimpleDoubleProperty(2.2);
    DoubleProperty exposureProperty = new SimpleDoubleProperty(0.18);
    
    DoubleProperty widthProperty = new SimpleDoubleProperty(0);
    DoubleProperty heightProperty = new SimpleDoubleProperty(0);
    
    private final TracerAPI api;
    
    public RenderDialog(TracerAPI api)
    {
        this.api = api;
        
        BorderPane box = initFXMLComponent();        
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
        
        gammaSlider.setValue(gammaProperty.doubleValue());
        gammaProperty.bind(gammaSlider.valueProperty());
        exposureSlider.setValue(exposureProperty.doubleValue());
        exposureProperty.bind(exposureSlider.valueProperty());
        
        
        convertToTextFieldDouble(gammaTextField);
        convertToTextFieldDouble(exposureTextField);

        bindBidirectionalStringAndDouble(gammaTextField.textProperty(), gammaSlider.valueProperty());  
        bindBidirectionalStringAndDouble(exposureTextField.textProperty(), exposureSlider.valueProperty());
        
        widthProperty.setValue(api.getImageWidth(MambaAPIInterface.ImageType.RENDER_IMAGE));
        heightProperty.setValue(api.getImageWidth(MambaAPIInterface.ImageType.RENDER_IMAGE));
        
        return box;
    }
    
}
