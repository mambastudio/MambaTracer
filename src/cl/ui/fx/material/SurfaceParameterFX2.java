/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.struct.CSurfaceParameter2;
import cl.ui.fx.Point3FX;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;
import jfx.dnd.ReadObjectsHelper;
import jfx.dnd.WriteObjectsHelper;

/**
 *
 * @author user
 */
public class SurfaceParameterFX2 implements Serializable {
    public transient BooleanProperty          textureTitledPaneExpanded;    
    public transient BooleanProperty          diffuseTitledPaneExpanded;    
    public transient BooleanProperty          glossyTitledPaneExpanded;    
    public transient BooleanProperty          mirrorTitledPaneExpanded;    
    public transient BooleanProperty          emissionTitledPaneExpanded;    
    
    //this surface is done by texture
    public transient ObjectProperty<ImageView>    diffuseTexture;
    public transient ObjectProperty<ImageView>    glossyTexture;
    public transient ObjectProperty<ImageView>    roughnessTexture;
    public transient ObjectProperty<ImageView>    mirrorTexture;
    
    //brdf parameters
    public transient ObjectProperty<Color>    diffuse_color;
    public transient Point3FX                 diffuse_param;
    public transient ObjectProperty<Color>    glossy_color;
    public transient Point3FX                 glossy_param;
    public transient ObjectProperty<Color>    mirror_color;
    public transient Point3FX                 mirror_param;
    public transient ObjectProperty<Color>    emission_color;
    public transient Point3FX                 emission_param;
    
    public SurfaceParameterFX2()
    {
        init();
    }
    
    public final void init()
    {
        textureTitledPaneExpanded   = new SimpleBooleanProperty(false);    
        diffuseTitledPaneExpanded   = new SimpleBooleanProperty(true);    
        glossyTitledPaneExpanded    = new SimpleBooleanProperty(true);    
        mirrorTitledPaneExpanded    = new SimpleBooleanProperty(false);    
        emissionTitledPaneExpanded  = new SimpleBooleanProperty(false);    

        diffuseTexture              = new SimpleObjectProperty(new ImageView());
        glossyTexture               = new SimpleObjectProperty(new ImageView());
        roughnessTexture            = new SimpleObjectProperty(new ImageView());
        mirrorTexture               = new SimpleObjectProperty(new ImageView());
        //isTexture.bind(texture.isNotNull());
        
        diffuse_color   = new SimpleObjectProperty(Color.web("#f2f2f2"));
        diffuse_param   = new Point3FX(1, 0, 0);
        glossy_color    = new SimpleObjectProperty(Color.web("#f2f2f2"));
        glossy_param    = new Point3FX(0, 0.01f, 0.01f);
        mirror_color    = new SimpleObjectProperty(Color.web("#f2f2f2"));
        mirror_param    = new Point3FX(0, -0.01f, 0);
        emission_color  = new SimpleObjectProperty(Color.web("#f2f2f2"));
        emission_param  = new Point3FX(0, 1f, 0);
    }
    
    public SurfaceParameterFX2 copy()
    {
        SurfaceParameterFX2 param = new SurfaceParameterFX2();
        
        param.textureTitledPaneExpanded.set(textureTitledPaneExpanded.get());
        param.diffuseTitledPaneExpanded.set(diffuseTitledPaneExpanded.get());
        param.glossyTitledPaneExpanded .set(glossyTitledPaneExpanded .get());
        param.mirrorTitledPaneExpanded .set(mirrorTitledPaneExpanded .get());
        param.emissionTitledPaneExpanded.set(emissionTitledPaneExpanded.get());
         
        param.diffuseTexture.set(diffuseTexture.get());
        param.glossyTexture.set(glossyTexture.get());
        param.roughnessTexture.set(roughnessTexture.get());
        param.mirrorTexture.set(mirrorTexture.get());
        
        param.diffuse_color.set(diffuse_color.get()); 
        param.diffuse_param.set(diffuse_param);  
        param.glossy_color.set(glossy_color.get());   
        param.glossy_param.set(glossy_param);   
        param.mirror_color.set(mirror_color.get());   
        param.mirror_param.set(mirror_param);            
        param.emission_color.set(emission_color.get()); 
        param.emission_param.set(emission_param); 
        return param;
    }
    
    public void set(SurfaceParameterFX2 param)
    {
        textureTitledPaneExpanded.set(param.textureTitledPaneExpanded.get());
        diffuseTitledPaneExpanded.set(param.diffuseTitledPaneExpanded.get());
        glossyTitledPaneExpanded.set(param.glossyTitledPaneExpanded.get());  
        mirrorTitledPaneExpanded.set(param.mirrorTitledPaneExpanded.get());  
        emissionTitledPaneExpanded.set(param.emissionTitledPaneExpanded.get());
        
        diffuseTexture.set(param.diffuseTexture.get());
        glossyTexture.set(param.glossyTexture.get());
        roughnessTexture.set(param.roughnessTexture.get());
        mirrorTexture.set(param.mirrorTexture.get());
        
        diffuse_color.set(param.diffuse_color.get()); 
        diffuse_param.set(param.diffuse_param);  
        glossy_color.set(param.glossy_color.get());   
        glossy_param.set(param.glossy_param);   
        mirror_color.set(param.mirror_color.get());   
        mirror_param.set(param.mirror_param);            
        emission_color.set(param.emission_color.get()); 
        emission_param.set(param.emission_param); 
        
    }
    
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
        WriteObjectsHelper.writeAllProp(s, 
                diffuseTitledPaneExpanded,  
                glossyTitledPaneExpanded,   
                mirrorTitledPaneExpanded,   
                emissionTitledPaneExpanded, 
                diffuseTexture,
                glossyTexture,
                roughnessTexture,
                mirrorTexture,
                diffuse_color,
                diffuse_param.x,
                diffuse_param.y,
                diffuse_param.z,
                glossy_color,
                glossy_param.x,
                glossy_param.y,
                glossy_param.z,
                mirror_color,
                mirror_param.x,
                mirror_param.y,
                mirror_param.z,
                emission_color,
                emission_param.x,
                emission_param.y,
                emission_param.z);
        
    }
    
    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        init();
        ReadObjectsHelper.readAllProp(s,
                diffuseTitledPaneExpanded,  
                glossyTitledPaneExpanded,   
                mirrorTitledPaneExpanded,   
                emissionTitledPaneExpanded, 
                diffuseTexture,
                glossyTexture,
                roughnessTexture,
                mirrorTexture,
                diffuse_color,
                diffuse_param.x,
                diffuse_param.y,
                diffuse_param.z,
                glossy_color,
                glossy_param.x,
                glossy_param.y,
                glossy_param.z,
                mirror_color,
                mirror_param.x,
                mirror_param.y,
                mirror_param.z,
                emission_color,
                emission_param.x,
                emission_param.y,
                emission_param.z);          
    }
        
    public void setSurfaceParameter(CSurfaceParameter2 param)
    {
        
        diffuse_color.set(param.diffuse_color.getColorFX());
        diffuse_param.set(param.diffuse_param);
        glossy_color.set(param.glossy_color.getColorFX());
        glossy_param.set(param.glossy_param);
        mirror_color.set(param.mirror_color.getColorFX());
        mirror_param.set(param.mirror_param);
        emission_color.set(param.emission_color.getColorFX());
        emission_param.set(param.emission_param);
        
    }
    
    public CSurfaceParameter2 getSurfaceParameter()
    {
        CSurfaceParameter2 param = new CSurfaceParameter2();
        
        param.isDiffuseTexture = diffuseTexture.getValue().getImage()!=null;
        param.isGlossyTexture = glossyTexture.getValue().getImage()!=null;
        param.isRoughnessTexture = roughnessTexture.getValue().getImage()!=null;
        param.isMirrorTexture = mirrorTexture.getValue().getImage()!=null;
        
        param.diffuse_color.setColorFX(diffuse_color.get());
        param.diffuse_param.setValue(diffuse_param.getCPoint3());
        param.glossy_color.setColorFX(glossy_color.get());
        param.glossy_param.setValue(glossy_param.getCPoint3());
        param.mirror_color.setColorFX(mirror_color.get());
        param.mirror_param.setValue(mirror_param.getCPoint3());
        param.emission_color.setColorFX(emission_color.get());
        param.emission_param.setValue(emission_param.getCPoint3());
        
        return param;
    }
    
    
}
