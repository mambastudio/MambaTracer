/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.brdf;

import cl.ui.mvc.viewmodel.MaterialEditorModel;
import coordinate.parser.attribute.MaterialT;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ColorPicker;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;

/**
 * FXML Controller class
 *
 * @author user
 */
public class BRDFEditorController implements Initializable {

    /**
     * Initializes the controller class.
     */
    
    @FXML
    TextField nameTextField;
    @FXML
    ColorPicker diffuseColorPicker;
    @FXML
    Slider diffuseWeightSlider;
    @FXML
    Label diffuseWeightLabel;
        
    @FXML
    ColorPicker reflectionColorPicker;
    @FXML
    Spinner<Double> exponentialU;
    @FXML
    Spinner<Double> exponentialV;
    @FXML
    Spinner<Double> iorSpinner;
    
    @FXML
    ColorPicker emitterColorPicker;
    @FXML
    Spinner<Double> emitterPowerSpinner;
    
    @FXML
    GridPane diffuseGridPane;
    @FXML
    GridPane reflectionGridPane;
    @FXML
    GridPane refractionGridPane;
    
    @FXML
    CheckBox refractionEnabled;
    @FXML
    CheckBox emitterEnabled;
    
    MaterialEditorModel editor = new MaterialEditorModel();
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        
        editor.registerNameTextField(nameTextField);
        
        editor.registerDiffuseColorPicker(diffuseColorPicker);
        editor.registerDiffuseWeightSlider(diffuseWeightSlider);
        editor.registerDiffuseWeightLabel(diffuseWeightLabel);
         
        editor.registerReflectionColorPicker(reflectionColorPicker);
        editor.registerExponentialUSpinner(exponentialU);
        editor.registerExponentialVSpinner(exponentialV);
        editor.registerRefractionEnabled(refractionEnabled);
        editor.registerIORSpinner(iorSpinner);
        
        editor.registerEmitterColorPicker(emitterColorPicker);
        editor.registerEmitterPowerSpinner(emitterPowerSpinner);
        editor.registerEmitterEnabled(emitterEnabled);
        
        editor.initMaterial(new MaterialT());
        
        //gui to gui component interaction
        diffuseGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        reflectionGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        refractionGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        emitterColorPicker.disableProperty().bind(emitterEnabled.selectedProperty().not());
        emitterPowerSpinner.disableProperty().bind(emitterEnabled.selectedProperty().not());
        
        iorSpinner.disableProperty().bind(refractionEnabled.selectedProperty().not());
        
        
    }    
    
    
    public void printMaterial(ActionEvent e)
    {
        System.out.println(editor.getEditedMaterial());
    }
    
}
