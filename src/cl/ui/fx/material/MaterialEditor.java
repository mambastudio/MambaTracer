/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.ui.fx.material.type.DiffuseFX;
import cl.abstracts.MaterialInterface.BRDFType;
import cl.fx.GalleryDialogFX;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ListView;
import javafx.scene.control.Separator;
import javafx.scene.image.Image;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import jfx.dialog.DialogAbstract;
import jfx.dialog.DialogUtility;
import jfx.form.PropertyNode;
import jfx.form.Setting;
import jfx.form.SimpleSetting;

/**
 *
 * @author user
 */
public class MaterialEditor extends DialogAbstract<MaterialFX> {
    @FXML
    VBox leftSide;
    @FXML
    VBox rightSide;
    @FXML
    ListView<SurfaceParameterFX> surfparamListView;
    @FXML
    ComboBox<BRDFType> brdfTypeCombo;
    @FXML
    StackPane brdfEditorPane;
    
    MaterialFX defMat;
    GalleryDialogFX dialog;
    
    ObjectProperty<String> opacityTypeSelection;
    ObservableList<String> opacityTypeList;
    
    public MaterialEditor(MaterialFX defMat, GalleryDialogFX dialogImages)
    {
        this.defMat = defMat.copy();
        this.dialog = dialogImages;
        
        init();
    }
    
    public final void init()
    {
        HBox box = new HBox();
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource(
            "MaterialEditor.fxml"));
        fxmlLoader.setRoot(box);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        } 
        
        this.setButtons(OK, CANCEL);    
        this.setContent(box);
        this.setSize(100, 500);
        
        this.setSupplier((buttonType)->{
            if(buttonType.equals(OK))
                return defMat;
            else
                return null;
        });
        
        //is portal
        PropertyNode isportal = SimpleSetting.createForm(
                Setting.of("Is Portal", defMat.isPortal));
        leftSide.getChildren().add(isportal);
        leftSide.getChildren().add(new Separator());
        
        //string name
        PropertyNode matName = SimpleSetting.createForm(
                Setting.of("Name", defMat.name));
        leftSide.getChildren().add(matName);
        leftSide.getChildren().add(new Separator());
                
        //opacity setup       
        initOpacity();
        
        //surface parameter list setup
        surfparamListView.getItems().addAll(defMat.param1, defMat.param2);
        surfparamListView.getSelectionModel().selectedItemProperty().addListener((obs, oV, nV)->{
            //change the list type in combobox based on what layer was selected
            int layerIndex = surfparamListView.getSelectionModel().getSelectedIndex();
            if(layerIndex == 0)
                brdfTypeCombo.getItems().setAll(BRDFType.values());
            else
                brdfTypeCombo.getItems().remove(BRDFType.EMITTER);
             
            
            brdfTypeCombo.setValue(BRDFType.values()[nV.brdfType.get()]);
            initBRDFEditor(surfparamListView.getSelectionModel().getSelectedItem(), BRDFType.values()[nV.brdfType.get()]);
        });
        
        //brdf type combobox setup
        brdfTypeCombo.disableProperty().bind(surfparamListView.getSelectionModel().selectedItemProperty().isNull());
        //setting up surface parameter type and editing
        brdfTypeCombo.getSelectionModel().selectedItemProperty().addListener((obs, oV, nV)->{  
            if(nV != null)
            {
                SurfaceParameterFX param = surfparamListView.getSelectionModel().getSelectedItem();
                param.brdfType.set(nV.ordinal());  //set brdf type (diffuse, emitter or... )   
                surfparamListView.refresh();
                initBRDFEditor(param, nV);
            }
        });
        
        
    }
    
    private void initOpacity()
    {
         //opacity setup       
        opacityTypeList = FXCollections.observableArrayList(
                Arrays.asList("Fresnel", "Texture", "Opacity"));
        opacityTypeSelection = new SimpleObjectProperty<>(
                opacityTypeList.get(defMat.opacityType.get())
        );
        //setup pane for opacity
        PropertyNode pane = SimpleSetting.createForm(Setting.of("Opacity Type", 
                        opacityTypeList, 
                        opacityTypeSelection),
                Setting.of("Fresnel", defMat.fresnel),
                Setting.of("Texture", Image.class, defMat.texture, ()->{
                    Scene sceneFX = this.getScene();
                    Optional<Path> path = DialogUtility.showAndWait(sceneFX, dialog);
                    if(path.isPresent())
                    {
                        try {
                            return new Image(path.get().toUri().toURL().toExternalForm());
                        } catch (MalformedURLException ex) {
                            Logger.getLogger(MaterialEditor.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }                    
                    return null;
                }),
                
                Setting.of("Opacity", defMat.opacity, 0, 1, 1, 0.1f)
        );
        
        opacityTypeSelection.addListener((obs, oldVal, newVal)->{
            opacityDisableSelection(pane, newVal);
        });        
        leftSide.getChildren().add(pane);
        
        //init
        opacityDisableSelection(pane, opacityTypeSelection.get());
       
       
    }
    
    private void opacityDisableSelection(PropertyNode pane, String type)
    {
        switch (type) {
            case "Opacity":
                pane.enableRow("Opacity");
                pane.disableRows("Texture", "Fresnel");
                defMat.opacityType.set(opacityTypeList.indexOf(type));
                break;
            case "Texture":
                pane.enableRow("Texture");
                pane.disableRows("Opacity", "Fresnel");
                defMat.opacityType.set(opacityTypeList.indexOf(type));
                break;
            case "Fresnel":
                pane.enableRow("Fresnel");
                pane.disableRows("Texture", "Opacity");
                defMat.opacityType.set(opacityTypeList.indexOf(type));
                break;
            default:
                break;
        }
    }
    
    
    public void initBRDFEditor(SurfaceParameterFX param, BRDFType type)
    {
        brdfEditorPane.getChildren().removeAll(brdfEditorPane.getChildren());
        if(type == BRDFType.DIFFUSE)
        {            
            DiffuseFX diffuse = new DiffuseFX();       
            diffuse.bindSurfaceParameter(param); //bind first
            brdfEditorPane.getChildren().add(diffuse.getPropertyNode());  //setup editor       
        }
        else if(type == BRDFType.EMITTER)
        {            
           
        }
    }
    
}
