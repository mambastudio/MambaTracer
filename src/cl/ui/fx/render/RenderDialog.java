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
import javafx.scene.layout.StackPane;
import cl.ui.fx.main.TracerAPI;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.event.ActionEvent;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import static jfx.util.BindingFX.bindBidirectionalStringAndDouble;
import static jfx.util.ImplUtils.convertToTextFieldDouble;

/**
 *
 * @author user
 */
public class RenderDialog extends StackPane{
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
    
    @FXML
    Label widthLabel;
    @FXML
    Label heightLabel;
    
    @FXML
    ProgressIndicator colormapIndicator;
    @FXML
    ProgressIndicator normalmapIndicator;
    @FXML
    ProgressIndicator opacitymapIndicator;
    
    @FXML
    ProgressIndicator denoiseIndicator;
        
    DoubleProperty gammaProperty;
    DoubleProperty exposureProperty = new SimpleDoubleProperty(0.18);
        
    private final TracerAPI api;
    private final StackPane programRootPane;
    
    public RenderDialog(StackPane programRootPane, TracerAPI api)
    {
        this.api = api;
        this.programRootPane = programRootPane;
        
        this.initFXMLComponent(); 
        this.renderPane.getChildren().add(api.getBlendDisplayGI());
    }
    
    
    private void initFXMLComponent()
    {        
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource(
            "RenderDialogFXML.fxml"));
        fxmlLoader.setRoot(this);
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
            programRootPane.getChildren().remove(this);
            api.setDevicePriority(RAYTRACE);
        });
        
        gammaProperty = new SimpleDoubleProperty(api.getDeviceGI().getGamma());
        exposureProperty = new SimpleDoubleProperty(api.getDeviceGI().getExposure());
        
        gammaSlider.setValue(gammaProperty.doubleValue());
        gammaProperty.bind(gammaSlider.valueProperty());
        exposureSlider.setValue(exposureProperty.doubleValue());
        exposureProperty.bind(exposureSlider.valueProperty());
        
        
        convertToTextFieldDouble(gammaTextField);
        convertToTextFieldDouble(exposureTextField);

        bindBidirectionalStringAndDouble(gammaTextField.textProperty(), gammaSlider.valueProperty());  
        bindBidirectionalStringAndDouble(exposureTextField.textProperty(), exposureSlider.valueProperty());
        
        widthLabel.setText(Integer.toString(api.getImageWidth(MambaAPIInterface.ImageType.RENDER_IMAGE)));
        heightLabel.setText(Integer.toString(api.getImageHeight(MambaAPIInterface.ImageType.RENDER_IMAGE)));
        
        gammaProperty.addListener((obs, oV, nV)->{
            api.getDeviceGI().setGamma(nV.floatValue());
        });
        
        exposureProperty.addListener((obs, oV, nV)->{
            api.getDeviceGI().setExposure(nV.floatValue());
        });
        
        colormapIndicator.setOpacity(0);
        normalmapIndicator.setOpacity(0);
        opacitymapIndicator.setOpacity(0);
        denoiseIndicator.setOpacity(0);        
    }
    
    public void resetFilmExposure(ActionEvent e)
    {        
        exposureSlider.setValue(0.18);
    }
    
    public void resetFilmGamma(ActionEvent e)
    {        
        gammaSlider.setValue(2.2);
    }
    
    public void resetRenderFrame(ActionEvent e)
    {
        api.getBlendDisplayGI().reset();
    }
}
