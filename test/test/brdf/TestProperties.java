/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.brdf;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;

/**
 *
 * @author user
 */
public class TestProperties {
    public ObjectProperty<String> value1;
    public ObjectProperty<Float> value2;
    
    public TestProperties()
    {
        value1 = new SimpleObjectProperty();
        value2 = new SimpleObjectProperty();
        bindValuesBidirectional();
        
        
    }
    
    public static void main(String... args)
    {
        TestProperties properties = new TestProperties();
        properties.value2.setValue(2f);
        System.out.println(properties.value1.get());
        properties.value1.set("3.0f");
        System.out.println(properties.value2.get());
    }
    
    public final void bindValuesBidirectional()
    {
        value1.addListener(new ChangeListener<String>(){
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value2.setValue(Float.valueOf(newValue));
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
        
        value2.addListener(new ChangeListener<Float>() {
            private boolean changing;
            
            @Override
            public void changed(ObservableValue<? extends Float> observable, Float oldValue, Float newValue) {
                if(!changing)
                {
                    try {
                        changing = true;
                        value1.setValue(newValue.toString()); 
                    }                        
                    finally {
                        changing = false;
                    }    
                }
            }
            
        });
    }
}
