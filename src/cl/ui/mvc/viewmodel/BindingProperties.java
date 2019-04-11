/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.viewmodel;

import coordinate.parser.attribute.Color4T;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.StringProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.scene.paint.Color;

/**
 *
 * @author user
 */
public class BindingProperties {
    public static void bindBidirectional(ObjectProperty<Color4T> value1, ObjectProperty<Color> value2)
    {
        value1.addListener(new ChangeListener<Color4T>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color4T> observable, Color4T oldValue, Color4T newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value2.setValue(new Color(newValue.r, newValue.g, newValue.b, 1));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Color>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color> observable, Color oldValue, Color newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value1.setValue(new Color4T((float)newValue.getRed(), (float)newValue.getGreen(), (float)newValue.getBlue()));                        
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }            
        });
    }
    
    public static void bindBidirectional(ObjectProperty<Double> value1, FloatProperty value2)
    { 
        value1.addListener(new ChangeListener<Number>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value2.setValue(newValue);
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Number>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value1.setValue(newValue.doubleValue());
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
        });
    }
    
    public static void bindBidirectional(DoubleProperty value1, StringProperty value2)
    {     
        value1.addListener(new ChangeListener<Number>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value2.setValue(String.format( "%.1f", newValue));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<String>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value1.setValue(Double.valueOf(newValue));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
        });
    }
    
    public static void bindBidirectional(FloatProperty value1, StringProperty value2)
    {     
        value1.addListener(new ChangeListener<Number>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value2.setValue(String.format( "%.1f", newValue));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<String>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value1.setValue(Double.valueOf(newValue));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
        });
    }
    
    public static void bindBidirectional(ObjectProperty<Double> value1, ObjectProperty<Color4T> value2, char value)
    {
        value1.addListener(new ChangeListener<Number>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value2.get().r = newValue.floatValue();
                                break;
                            case 'g':
                                value2.get().g = newValue.floatValue();
                                break;
                            case 'b':
                                value2.get().b = newValue.floatValue();
                                break;
                            case 'w':
                                value2.get().w = newValue.floatValue();
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Color4T>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color4T> observable, Color4T oldValue, Color4T newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value1.setValue((double)newValue.r);
                                break;
                            case 'g':
                                value1.setValue((double)newValue.g); 
                                break;
                            case 'b':
                                value1.setValue((double)newValue.b);
                                break;
                            case 'w':
                                value1.setValue((double)newValue.w);
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }            
        });
    }
    
    
    public static void bindBidirectional(DoubleProperty value1, ObjectProperty<Color4T> value2, char value)
    {
        value1.addListener(new ChangeListener<Number>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value2.get().r = newValue.floatValue();
                                break;
                            case 'g':
                                value2.get().g = newValue.floatValue();
                                break;
                            case 'b':
                                value2.get().b = newValue.floatValue();
                                break;
                            case 'w':
                                value2.get().w = newValue.floatValue();
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Color4T>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color4T> observable, Color4T oldValue, Color4T newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value1.setValue((double)newValue.r);
                                break;
                            case 'g':
                                value1.setValue((double)newValue.g); 
                                break;
                            case 'b':
                                value1.setValue((double)newValue.b);
                                break;
                            case 'w':
                                value1.setValue((double)newValue.w);
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }            
        });
    }
    
     
    public static void bindBidirectional(BooleanProperty value1, ObjectProperty<Color4T> value2, char value)
    {
        value1.addListener(new ChangeListener<Boolean>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
                if(!changing)
                { 
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value2.get().r = newValue ? 1 : 0;
                                break;
                            case 'g':
                                value2.get().g = newValue ? 1 : 0;
                                break;
                            case 'b':
                                value2.get().b = newValue ? 1 : 0;
                                break;
                            case 'w':
                                value2.get().w = newValue ? 1 : 0;
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Color4T>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color4T> observable, Color4T oldValue, Color4T newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value1.setValue(newValue.r == 1);
                                break;
                            case 'g':
                                value1.setValue(newValue.g == 1); 
                                break;
                            case 'b':
                                value1.setValue(newValue.b == 1);
                                break;
                            case 'w':
                                value1.setValue(newValue.w == 1);
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }            
        });
    }
    
    public static void bindBidirectional(StringProperty value1, ObjectProperty<Color4T> value2, char value)
    {
        value1.addListener(new ChangeListener<String>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value2.get().r = Float.parseFloat(newValue);
                                break;
                            case 'g':
                                value2.get().g = Float.parseFloat(newValue);
                                break;
                            case 'b':
                                value2.get().b = Float.parseFloat(newValue);
                                break;
                            case 'w':
                                value2.get().w = Float.parseFloat(newValue);
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Color4T>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Color4T> observable, Color4T oldValue, Color4T newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        switch (value) {
                            case 'r':
                                value1.setValue(Float.toString(newValue.r));
                                break;
                            case 'g':
                                value1.setValue(Float.toString(newValue.g)); 
                                break;
                            case 'b':
                                value1.setValue(Float.toString(newValue.b));
                                break;
                            case 'w':
                                value1.setValue(Float.toString(newValue.w));
                                break;
                            default:
                                break;
                        }
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }            
        });
    }
}
