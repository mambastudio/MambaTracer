/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc;

import static cl.ui.mvc.CustomData.Type.LEAF;
import java.io.Serializable;
import javafx.scene.input.DataFormat;

/**
 *
 * @author user
 * @param <Q>
 */
public class CustomData<Q> implements Serializable {
    private static final DataFormat format = new DataFormat("CUSTOMDATA");
    
    private String name;
    private Q q;
    private Type t;
    
    public enum Type{MATERIAL, PARENT, LEAF};
    
    public CustomData(String name, Q q)
    {
        this.name = name;
        this.q = q;
        this.t = LEAF;
    }
    
    public CustomData(String name, Q q, Type t)
    {
        this.name = name;
        this.q = q;
        this.t = t;
    }
    
    public Type getType()
    {
        return t;
    }
    
    public static DataFormat getFormat()
    {
        return format;
    }
    
    public String getName()
    {
        return name;
    }
    
    public void setName(String name)
    {
        this.name = name;
    }
        
    public Q getData()
    {
        return q;
    }
    
    public void setData(Q q)
    {
        this.q = q;
    }
}
