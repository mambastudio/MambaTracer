/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.viewmodel;

import coordinate.parser.attribute.Color4T;
import coordinate.parser.attribute.MaterialT;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleFloatProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ColorPicker;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.TextField;

/**
 *
 * @author user
 */
public class MaterialEditorModel {
    //Name
    public StringProperty name;
    //Diffuse
    public ObjectProperty<Color4T> diffuse;
    public FloatProperty diffuseWeight;

    //Reflection and refraction
    public ObjectProperty<Color4T> reflection;
    public FloatProperty eu, ev, ior;
    public BooleanProperty iorEnabled;
      
    //Emitter
    public ObjectProperty<Color4T> emitter;
    public BooleanProperty emitterEnabled;
    
    private MaterialT material = null;
    
    
    public MaterialEditorModel()
    {
        material = new MaterialT();
        
        name = new SimpleStringProperty();
        
        diffuse = new SimpleObjectProperty<>();
        diffuseWeight = new SimpleFloatProperty();
          
        reflection = new SimpleObjectProperty<>();
        eu = new SimpleFloatProperty();
        ev = new SimpleFloatProperty();
        ior = new SimpleFloatProperty();
        iorEnabled = new SimpleBooleanProperty();
        
        emitter = new SimpleObjectProperty<>();
        emitterEnabled = new SimpleBooleanProperty();
        
        registerListeners();
    }
    
    private void registerListeners()
    {
        name.addListener((o, oldValue, newValue) -> {
            material.name = newValue;
        });
        diffuse.addListener((o, oldValue, newValue) -> {
            material.diffuse = newValue.copy();
        });
        diffuseWeight.addListener((o, oldValue, newValue) -> {
            material.diffuseWeight = newValue.floatValue();
        });
        reflection.addListener((o, oldValue, newValue) -> {
            material.reflection = newValue.copy();
        });
        eu.addListener((o, oldValue, newValue) -> {
            material.eu = newValue.floatValue();
        });
        ev.addListener((o, oldValue, newValue) -> {
            material.ev = newValue.floatValue();
        });
        ior.addListener((o, oldValue, newValue) -> {
            material.ior = newValue.floatValue();
        });
        iorEnabled.addListener((o, oldValue, newValue) -> {
            material.iorEnabled = newValue;
        });
        emitter.addListener((o, oldValue, newValue) -> {
            material.emitter = newValue.copy();
        });
        emitterEnabled.addListener((o, oldValue, newValue) -> {
            material.emitterEnabled = newValue;
        });
    }
    
    public void initMaterial(MaterialT material)
    {      
        name.setValue(material.name);
        
        diffuse.setValue(material.diffuse);       
        diffuseWeight.setValue(material.diffuseWeight);
        
        reflection.setValue(material.reflection);
        eu.setValue(material.eu);
        ev.setValue(material.ev);
        ior.setValue(material.ior);
        iorEnabled.setValue(material.iorEnabled);
        
        emitter.setValue(material.emitter);
        emitterEnabled.setValue(material.emitterEnabled);        
    }   
  
    public void registerNameTextField(TextField nameTextField)
    {
        name.bindBidirectional(nameTextField.textProperty());
    }
    
    public void registerDiffuseColorPicker(ColorPicker diffuseColorPicker)
    {        
        BindingProperties.bindBidirectional(diffuse, diffuseColorPicker.valueProperty());
    }
    
    public void registerDiffuseWeightSlider(Slider diffuseWeightSlider)
    {
        diffuseWeightSlider.valueProperty().bindBidirectional(diffuseWeight);               
    }
    
    public void registerDiffuseWeightLabel(Label diffuseWeightLabel)
    {
        BindingProperties.bindBidirectional(diffuseWeight, diffuseWeightLabel.textProperty());
    }
        
    public void registerReflectionColorPicker(ColorPicker reflectionColorPicker)
    {        
        BindingProperties.bindBidirectional(reflection, reflectionColorPicker.valueProperty());
    }
    
    public void registerExponentialUSpinner(Spinner<Double> spinner)
    {
        SpinnerValueFactory<Double> factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 0, 10);
        spinner.setValueFactory(factory);
        BindingProperties.bindBidirectional(factory.valueProperty(), eu);
    }
    
    public void registerExponentialVSpinner(Spinner<Double> spinner)
    {
        SpinnerValueFactory<Double> factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 0, 10);
        spinner.setValueFactory(factory);
        BindingProperties.bindBidirectional(factory.valueProperty(), ev);
    }
    
    public void registerRefractionEnabled(CheckBox refractionEnabled)
    {
        refractionEnabled.selectedProperty().bindBidirectional(iorEnabled);
    }
        
    public void registerIORSpinner(Spinner<Double> spinner)
    {
        SpinnerValueFactory<Double> factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1, 4, 1, 0.5);
        spinner.setValueFactory(factory);
        BindingProperties.bindBidirectional(factory.valueProperty(), ior);
    }
    
    public void registerEmitterEnabled(CheckBox emitterEnabled)
    {
        emitterEnabled.selectedProperty().bindBidirectional(this.emitterEnabled);
    }
    
    public void registerEmitterColorPicker(ColorPicker emitterColorPicker)
    {        
        BindingProperties.bindBidirectional(emitter, emitterColorPicker.valueProperty());
    }
    
    public void registerEmitterPowerSpinner(Spinner<Double> emitterSpinner)
    {
        SpinnerValueFactory<Double> factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1, 100, 5, 5);
        emitterSpinner.setValueFactory(factory);
        BindingProperties.bindBidirectional(factory.valueProperty(), emitter, 'w');
    }
    
    public MaterialT getEditedMaterial()
    {
        return material.copy();
    }
}
