/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.abstracts.MaterialInterface;
import cl.struct.CMaterial2;
import coordinate.parser.attribute.MaterialT;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import jfx.dnd.ReadObjectsHelper;
import jfx.dnd.WriteObjectsHelper;

/**
 *
 * @author user
 */
public class MaterialFX2 implements Serializable, MaterialInterface<MaterialFX2>{
    //surface
    public transient SurfaceParameterFX2 param; 
    public transient StringProperty name;    
    private transient CMaterial2 cmaterial;
    
    public MaterialFX2(String name)
    {
        this();
        this.name.set(name);
    }
    
    public MaterialFX2()
    {
        param = new SurfaceParameterFX2();        
        name = new SimpleStringProperty("default");
    }
    
    @Override
    public Image getDiffuseTexture()
    {
        if(param.diffuseTexture.get() != null)
            return param.diffuseTexture.get().getImage();
        return null;
    }
    
    @Override
    public Image getGlossyTexture()
    {
        if(param.glossyTexture.get() != null)
            return param.glossyTexture.get().getImage();
        return null;
    }
    
    @Override
    public Image getRoughnessTexture()
    {
        if(param.roughnessTexture.get() != null)
            return param.roughnessTexture.get().getImage();
        return null;
    }
    
    public final void init()
    {
        param = new SurfaceParameterFX2();        
        name = new SimpleStringProperty("default");
    }
    
    public void setCMaterial(CMaterial2 material)
    {
        this.cmaterial = material;        
        param.setSurfaceParameter(material.param);        
    }
    
    public void refreshCMaterial()
    {
        if(cmaterial != null)
        {
            cmaterial.setSurfaceParameter(param.getSurfaceParameter());
            cmaterial.refreshGlobalArray();        
        }
            
    }

    @Override
    public void setMaterial(MaterialFX2 m) {
        param.set(m.param);
        name.set(m.name.get());
        refreshCMaterial();        
    }

    @Override
    public MaterialFX2 copy() {
        MaterialFX2 mat = new MaterialFX2();
        mat.param.set(param);
        mat.name.set(name.get());
        return mat;
    }
    
    public void setDiffuseColor(float r, float g, float b)
    {
        float r_ = getRange(r, 0, 1);
        float g_ = getRange(g, 0, 1);
        float b_ = getRange(b, 0, 1);
        param.diffuse_color.set(Color.color(r_, g_, b_));  
        refreshCMaterial();
    }
    
    public void setDiffuseAmount(float value)
    {
        float level = getRange(value, 0, 1);        
        param.diffuse_param.x.set(level);
        refreshCMaterial();      
    }
    
    public void setGlossyColor(float r, float g, float b)
    {
        float r_ = getRange(r, 0, 1);
        float g_ = getRange(g, 0, 1);
        float b_ = getRange(b, 0, 1);
        param.glossy_color.set(Color.color(r_, g_, b_));  
        refreshCMaterial();
    }
    
    public void setGlossyAmount(float value)
    {
        float level = getRange(value, 0, 1);        
        param.glossy_param.x.set(level);
        refreshCMaterial();      
    }
    
    public void setGlossRoughness(float ax, float ay)
    {
        float ax_ = getRange(ax, 0, 1);    
        float ay_ = getRange(ay, 0, 1);
        param.glossy_param.y.set(ax_);
        param.glossy_param.z.set(ay_);
        refreshCMaterial();      
    }
    
    public void setGlossRoughness(float a)
    {
        float a_ = getRange(a, 0, 1);           
        param.glossy_param.y.set(a_);
        param.glossy_param.z.set(a_);
        refreshCMaterial();      
    }
    
    private float getRange(float value, float min, float max)
    {
        float min_, max_;
        
        if(min < max)
        {
            min_ = min;
            max_ = max;
        }
        else if(max < min)
        {
            min_ = max;
            max_ = min;
        }
        else
        {
            return value;
        }        
        
        float level = value;
        if(level <= min_) level = min_;
        if(level >= max_) level = max_;
        
        return level;
    }

    @Override
    public void setMaterialT(MaterialT t) {
        param.diffuse_color.set(new Color(t.diffuse.r, t.diffuse.g, t.diffuse.b, 1));
    }
    
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
        s.writeObject(param);        
        WriteObjectsHelper.writeAllProp(s, 
                name);
    }

    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        init();      
        param = (SurfaceParameterFX2) s.readObject();        
        ReadObjectsHelper.readAllProp(s, 
                name);
        
    }
    
    @Override
    public String toString()
    {
        return name.get();
    }
}
