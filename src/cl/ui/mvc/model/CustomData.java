/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.model;

import cl.core.CMaterialInterface;
import coordinate.parser.attribute.MaterialT;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.scene.input.DataFormat;

/**
 *
 * @author user
 * @param <Q>
 * 
 * https://stackoverflow.com/questions/18791566/notserializableexception-on-simplelistproperty
 * https://gist.github.com/james-d/a7202039b00170256293
 * 
 */
public class CustomData<Q> implements Serializable, CMaterialInterface {
    private static final DataFormat format = new DataFormat("CUSTOMDATA");  
    
    private transient StringProperty name = new SimpleStringProperty();
    private transient ObjectProperty<Q> q = new SimpleObjectProperty();
    
    @Override
    public void setMaterial(MaterialT mat) {
        if(q.get() instanceof MaterialT)
        {
            this.name.setValue(mat.name);
            this.q.setValue((Q)mat);
        }
    }

    @Override
    public MaterialT getMaterial() {
        if(q.get() instanceof MaterialT)
            return (MaterialT)q.get();
        else
            return null;
    }
    
    public CustomData(String name, Q q)
    {
        this.name.setValue(name);
        this.q.setValue(q);
    }
    
    public StringProperty getNameProperty()
    {
        return name;
    }
    
    public ObjectProperty<Q> getQProperty()
    {
        return q;
    }
               
    public static DataFormat getFormat()
    {
        return format;
    }
    
    public String getName()
    {
        return name.get();
    }
    
    public void setName(String name)
    {
        this.name.setValue(name);
    }
        
    public Q getData()
    {
        return q.get();
    }
    
    public void setData(Q q)
    {
        this.q.setValue(q);        
    }
    
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
        s.writeUTF(name.get());
        s.writeObject(q.get());
    }
    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        name = new SimpleStringProperty(s.readUTF());
        q = new SimpleObjectProperty(s.readObject());
    }
}
