/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.struct.CMaterial;
import cl.abstracts.MaterialInterface;
import coordinate.parser.attribute.MaterialT;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleFloatProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.scene.image.Image;
import jfx.dnd.ReadObjectsHelper;
import jfx.dnd.WriteObjectsHelper;

/**
 *
 * @author user
 */
public class MaterialFX implements Serializable, MaterialInterface<MaterialFX>{
    public transient SurfaceParameterFX param1; //surface 1
    public transient SurfaceParameterFX param2; //surface 2
    
    //name
    public transient StringProperty name;
    
    //portal?
    public transient BooleanProperty isPortal;
    
    //either fresnel, texture, or value = (0, 1, 2);
    public transient IntegerProperty opacityType;
    
    public transient FloatProperty fresnel;
    public transient ObjectProperty<Image> texture;
    public transient FloatProperty opacity;
    
    private transient CMaterial material = null;
    
    public MaterialFX(String name)
    {
        this();
        this.name.set(name);
    }
    
    public MaterialFX()
    {
        param1 = new SurfaceParameterFX();
        param2 = new SurfaceParameterFX();
        
        name = new SimpleStringProperty("default");
        
        isPortal = new SimpleBooleanProperty(false);
        opacityType = new SimpleIntegerProperty(0);
        
        fresnel = new SimpleFloatProperty(1.2f);
        texture = new SimpleObjectProperty(null);
        opacity = new SimpleFloatProperty(1);
    }
    
    public final void init()
    {
        param1 = new SurfaceParameterFX();
        param2 = new SurfaceParameterFX();
        
        name = new SimpleStringProperty("default");
        
        isPortal = new SimpleBooleanProperty(false);
        opacityType = new SimpleIntegerProperty(0);
        
        fresnel = new SimpleFloatProperty(1.2f);
        texture = new SimpleObjectProperty(null);
        opacity = new SimpleFloatProperty(1);
    }
    
    @Override
    public MaterialFX copy()
    {
        MaterialFX materialFX = new MaterialFX();
        materialFX.param1.set(param1);
        materialFX.param2.set(param2);
        materialFX.name.set(name.get());
        materialFX.isPortal.set(isPortal.get());
        materialFX.opacityType.set(opacityType.get());
        materialFX.fresnel.set(fresnel.get());
        materialFX.texture.set(texture.get());
        materialFX.opacity.set(opacity.get());
        
        return materialFX;
    }
    
    @Override
    public void setMaterial(MaterialFX matFx)
    {
        param1.set(matFx.param1);
        param2.set(matFx.param2);
        name.set(matFx.name.get());
        isPortal.set(matFx.isPortal.get());
        opacityType.set(matFx.opacityType.get());
        fresnel.set(matFx.fresnel.get());
        texture.set(matFx.texture.get());
        opacity.set(matFx.opacity.get());
        
        refreshCMaterial();
    }
    
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
        s.writeObject(param1);
        s.writeObject(param2);
        WriteObjectsHelper.writeAllProp(s, 
                name, 
                isPortal,
                opacityType,
                fresnel,
                opacity);
    }

    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        init();      
        param1 = (SurfaceParameterFX) s.readObject();
        param2 = (SurfaceParameterFX) s.readObject();        
        ReadObjectsHelper.readAllProp(s, 
                name, 
                isPortal,
                opacityType,
                fresnel,
                opacity);
        
    }
    
    public void setCMaterial(CMaterial material)
    {
        this.material = material;        
        param1.setSurfaceParameter(material.param1);
        param2.setSurfaceParameter(material.param2);
        isPortal.set(material.isPortal);
        opacityType.set(material.opacityType);
        fresnel.set(material.fresnel);
        opacity.set(material.opacity);        
    }
    
    public void refreshCMaterial()
    {
        if(material != null)
        {
            material.setSurfaceParameter(param1.getSurfaceParameter(), param2.getSurfaceParameter());
            material.isPortal = isPortal.get();
            material.opacityType = opacityType.get();
            material.fresnel = fresnel.get();
            material.opacity = opacity.get();
            
            material.refreshGlobalArray();
        }
            
    }
    
    public CMaterial getCopySMaterial()
    {
        CMaterial mat = new CMaterial();
        return mat;
    }
    
    @Override
    public String toString()
    {
        return name.get();
    }

    @Override
    public void setMaterialT(MaterialT t) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
