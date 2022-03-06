/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import java.io.File;
import java.net.MalformedURLException;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.geometry.Insets;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TitledPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import jfx.form.Main;
import jfx.form.PropertyNode;
import jfx.form.Setting;
import jfx.form.SimpleSetting;

/**
 *
 * @author user
 */
public class SurfaceParameterFX2Editor extends ScrollPane {
    private final SurfaceParameterFX2 param;
    
    FileChooser fc = new FileChooser();
    Supplier<ImageView> fileImageExplorer;
    
    public SurfaceParameterFX2Editor(SurfaceParameterFX2 param)
    {
        super();
        
        //functional interface for file image exploration
        //call it before init(); 
        fileImageExplorer = ()->{                        
            //filter
            File selectedFile = fc.showOpenDialog(null);

            if(selectedFile != null)
            {
                try {
                    return new ImageView(new Image(selectedFile.toURI().toURL().toExternalForm()));
                } catch (MalformedURLException ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            return null;
        };
        
        this.param = param;
        init();
        
        //file filter
        FileChooser.ExtensionFilter fileExtensions = 
            new FileChooser.ExtensionFilter("Image formats", "*.jpg", "*.png");
        fc.getExtensionFilters().add(fileExtensions);
        
        
    }
    
    private void init()
    {
        if(param != null)
        {
            TitledPane diffuseTP = new TitledPane("diffuse"  , new VBox(SimpleSetting.createForm(                      
                    Setting.of("level", param.diffuse_param.getXProperty(), 0.0f, 1f, 0.00f, 0.01f),
                    Setting.of("diffuse color", Color.class, param.diffuse_color),
                    Setting.of("image", ImageView.class, param.diffuseTexture, fileImageExplorer))));
            
            TitledPane glossyTP = new TitledPane("glossy", new VBox(SimpleSetting.createForm(                    
                    Setting.of("level", param.glossy_param.getXProperty(), 0.0f, 1f, 0.00f, 0.01f),                    
                    Setting.of("glossy color", Color.class, param.glossy_color),      
                    Setting.of("glossy image", ImageView.class, param.glossyTexture, fileImageExplorer),
                    Setting.of("rough image", ImageView.class, param.roughnessTexture, fileImageExplorer),
                    Setting.of("anisotropic x", param.glossy_param.getYProperty(), 0.00001f, 1f, 0.00001f, 0.00001f),
                    Setting.of("anisotropic y", param.glossy_param.getZProperty(), 0.00001f, 1f, 0.00001f, 0.00001f))));
            TitledPane mirrorTP = new TitledPane("mirror", new VBox(SimpleSetting.createForm(                    
                    Setting.of("level", param.mirror_param.getXProperty(), 0.0f, 1f, 0.00f, 0.01f),
                    Setting.of("mirror color", Color.class, param.mirror_color),
                    Setting.of("ior", param.mirror_param.getYProperty(), -0.01f, 3f, -0.01f, 0.01f))));
            TitledPane emissionTP = new TitledPane("emission", new VBox(SimpleSetting.createForm( 
                    Setting.of("is present", param.emission_param.getXProperty(), false),
                    Setting.of("power", param.emission_param.getYProperty(), 1f, 50f, 0.01f, 0.01f))));
            
            //set first
            diffuseTP.setExpanded(param.diffuseTitledPaneExpanded.get());
            glossyTP.setExpanded(param.glossyTitledPaneExpanded.get());
            mirrorTP.setExpanded(param.mirrorTitledPaneExpanded.get());
            emissionTP.setExpanded(param.emissionTitledPaneExpanded.get());
            
            //bind (one direction or bidirectional)
            param.diffuseTitledPaneExpanded.bindBidirectional(diffuseTP.expandedProperty());
            param.glossyTitledPaneExpanded.bindBidirectional(glossyTP.expandedProperty());
            param.mirrorTitledPaneExpanded.bindBidirectional(mirrorTP.expandedProperty());
            param.emissionTitledPaneExpanded.bindBidirectional(emissionTP.expandedProperty());
            
            VBox vbox = new VBox(diffuseTP, glossyTP, mirrorTP, emissionTP);
            vbox.setSpacing(3);
            
            setContent(vbox);
        }
        this.setPadding(new Insets(3, 3, 3, 3));
        this.setVbarPolicy(ScrollBarPolicy.ALWAYS);
        //setStyle("-fx-background-color:transparent;");
    }
    
    public PropertyNode getPropertyNode()
    {
        return (PropertyNode) this.getContent();
    }
}
