/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.struct.CSurfaceParameter;
import cl.abstracts.MaterialInterface.BRDFType;
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
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import jfx.dnd.ReadObjectsHelper;
import jfx.dnd.WriteObjectsHelper;

/**
 *
 * @author user
 */
public class SurfaceParameterFX implements Serializable {
    //this surface is done by texture
    public transient BooleanProperty          isTexture;
    public transient IntegerProperty          numTexture;
    public transient ObjectProperty<Image>    texture;
    
    //what brdf
    public transient IntegerProperty          brdfType;
            
    //brdf parameters
    public transient ObjectProperty<Color>    base_color;
    public transient FloatProperty            diffuse_roughness;
    public transient ObjectProperty<Color>    specular_color;
    public transient FloatProperty            specular_IOR;
    public transient FloatProperty            metalness;
    public transient FloatProperty            transmission;
    public transient ObjectProperty<Color>    transmission_color;
    public transient FloatProperty            emission;
    public transient ObjectProperty<Color>    emission_color;
    public transient FloatProperty            anisotropy_nx;
    public transient FloatProperty            anisotropy_ny;
    
    public SurfaceParameterFX()
    {         
        init();
    }
    
    public final void init()
    {
        isTexture = new SimpleBooleanProperty(false);
        numTexture = new SimpleIntegerProperty(0);
        texture = new SimpleObjectProperty(null);

        brdfType = new SimpleIntegerProperty(0);


        base_color = new SimpleObjectProperty(Color.web("#f2f2f2"));
        diffuse_roughness = new SimpleFloatProperty(0);
        specular_color = new SimpleObjectProperty(Color.web("#f2f2f2"));
        specular_IOR = new SimpleFloatProperty(0);
        metalness = new SimpleFloatProperty(0);
        transmission = new SimpleFloatProperty(0);
        transmission_color = new SimpleObjectProperty(Color.web("#f2f2f2"));
        emission = new SimpleFloatProperty(20);
        emission_color = new SimpleObjectProperty(Color.WHITE);
        anisotropy_nx = new SimpleFloatProperty(0);
        anisotropy_ny = new SimpleFloatProperty(0);
    }
    
    public SurfaceParameterFX copy()
    {
        SurfaceParameterFX param = new SurfaceParameterFX();
        param.isTexture.set(isTexture.get());       
        param.numTexture.set(numTexture.get());
        param.texture.set(texture.get());
        param.brdfType.set(brdfType.get());
        param.base_color.set(base_color.get());
        param.diffuse_roughness.set(diffuse_roughness.get());
        param.specular_color.set(specular_color.get());
        param.specular_IOR.set(specular_IOR.get());
        param.metalness.set(metalness.get());
        param.transmission.set(transmission.get());
        param.transmission_color.set(transmission_color.get());
        param.emission.set(emission.get());
        param.emission_color.set(emission_color.get());
        param.anisotropy_nx.set(anisotropy_nx.get());
        param.anisotropy_ny.set(anisotropy_ny.get());
        return param;
    }
    
    public void set(SurfaceParameterFX param)
    {
        isTexture.set(param.isTexture.get());       
        numTexture.set(param.numTexture.get());
        texture.set(param.texture.get());
        brdfType.set(param.brdfType.get()); 
        base_color.set(param.base_color.get());
        diffuse_roughness.set(param.diffuse_roughness.get());
        specular_color.set(param.specular_color.get());
        specular_IOR.set(param.specular_IOR.get());
        metalness.set(param.metalness.get());
        transmission.set(param.transmission.get());
        transmission_color.set(param.transmission_color.get());
        emission.set(param.emission.get());
        emission_color.set(param.emission_color.get());
        anisotropy_nx.set(param.anisotropy_nx.get());
        anisotropy_ny.set(param.anisotropy_ny.get());
    }
    
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
        WriteObjectsHelper.writeAllProp(s, 
                isTexture,
                numTexture,               
                brdfType,
                base_color,
                diffuse_roughness,
                specular_color,
                specular_IOR,
                metalness,
                transmission,
                transmission_color,
                emission,
                emission_color,
                anisotropy_nx,
                anisotropy_ny);        
        
    }

    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        init();
        ReadObjectsHelper.readAllProp(s, 
                isTexture,
                numTexture,
                brdfType,
                base_color,
                diffuse_roughness,
                specular_color,
                specular_IOR,
                metalness,
                transmission,
                transmission_color,
                emission,
                emission_color,
                anisotropy_nx,
                anisotropy_ny);    
    }
    
    @Override
    public String toString()
    {
        return BRDFType.values()[brdfType.get()].toString();
    }
    
    public void setSurfaceParameter(CSurfaceParameter param)
    {
        isTexture.set(param.isTexture);
        numTexture.set(param.numTexture);
       
        brdfType.set(param.brdfType);

        base_color.set(param.base_color.getColorFX());
        diffuse_roughness.set(param.diffuse_roughness);
        specular_color.set(param.specular_color.getColorFX());
        specular_IOR.set(param.specular_IOR);
        metalness.set(param.metalness);
        transmission.set(param.transmission);
        transmission_color.set(param.transmission_color.getColorFX());
        emission.set(param.emission); 
        emission_color.set(param.emission_color.getColorFX());
        anisotropy_nx.set(param.anisotropy_nx);
        anisotropy_ny.set(param.anisotropy_ny);
    }
    
    public CSurfaceParameter getSurfaceParameter()
    {
        CSurfaceParameter param = new CSurfaceParameter();
        param.isTexture             = isTexture.get();
        param.numTexture            = numTexture.get();

        param.brdfType              = brdfType.get();
        
        param.base_color.setColorFX(base_color.get());  
        param.diffuse_roughness     =diffuse_roughness.get();
        param.specular_color.setColorFX(specular_color.get());
        param.specular_IOR          =specular_IOR .get();
        param.metalness             =metalness .get();
        param.transmission          =transmission .get();
        param.transmission_color.setColorFX(transmission_color .get());
        param.emission              =emission.get();
        param.emission_color.setColorFX(emission_color .get());
        param.anisotropy_nx         =anisotropy_nx.get();
        param.anisotropy_ny         =anisotropy_ny.get();
        
        return param;
    }
}
