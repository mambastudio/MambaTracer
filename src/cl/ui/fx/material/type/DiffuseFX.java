/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material.type;

import cl.abstracts.MaterialInterface.BRDFType;
import static cl.abstracts.MaterialInterface.BRDFType.DIFFUSE;
import cl.fx.GalleryDialogFX;
import cl.fx.UtilityHandler;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import jfx.dialog.DialogUtility;
import jfx.form.PropertyNode;
import jfx.form.Setting;
import jfx.form.SimpleSetting;
import cl.ui.fx.material.SurfaceParameterFX;

/**
 *
 * @author user
 */
public class DiffuseFX implements AbstractBrdfFX{

    public ObjectProperty<Color>    base_color;
    public ObjectProperty<Image>    texture;
    
    public DiffuseFX()
    {
        base_color = new SimpleObjectProperty();
        texture = new SimpleObjectProperty();
    }
    
    @Override
    public BRDFType getType() {
        return DIFFUSE;
    }

    @Override
    public PropertyNode getPropertyNode() {
        return  SimpleSetting.createForm(Setting.of("Base color", Color.class, base_color),
                Setting.of("Texture", Image.class, texture, ()->{
                    GalleryDialogFX dialogGallery = UtilityHandler.getGallery("texture");
                    Scene scene = UtilityHandler.getScene();
                    Optional<Path> path = DialogUtility.showAndWait(scene, dialogGallery);
                    if(path.isPresent())
                    {
                        try {
                            return new Image(path.get().toUri().toURL().toExternalForm());
                        } catch (MalformedURLException ex) {
                            Logger.getLogger(DiffuseFX.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }                    
                    return null;                    
                })
        );
    }

    @Override
    public void bindSurfaceParameter(SurfaceParameterFX param) {
        //good to unbind first
        param.base_color.unbind();      
        base_color.setValue(param.base_color.get());
        param.base_color.bind(base_color);
        
        //param.brdfType
        
        param.texture.unbind();
        texture.setValue(param.texture.get());
        param.texture.bind(texture);
        
        param.isTexture.unbind();
        param.isTexture.bind(texture.isNotNull());
        
        
    }
    
}
