/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.main;

import cl.struct.CRay;
import cl.struct.CBound;
import bitmap.image.BitmapARGB;
import bitmap.image.BitmapRGBE;
import bitmap.reader.HDRBitmapReader;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.abstracts.MambaAPIInterface.DeviceType.RENDER;
import cl.abstracts.MambaAPIInterface.ImageType;
import static cl.abstracts.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import cl.abstracts.RenderControllerInterface;
import cl.data.CPoint3;
import cl.data.CVector3;
import cl.device.CDeviceRT;
import cl.fx.UtilityHandler;
import cl.ui.fx.material.MaterialFX;
import cl.ui.fx.render.RenderDialog;
import cl.ui.fx.TreeCellMaterialDestinationFX;
import cl.ui.fx.TreeCellMaterialSourceFX;
import coordinate.model.OrientationModel;
import coordinate.parser.attribute.MaterialT;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Point2D;
import javafx.scene.control.Button;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.StackPane;
import jfx.dialog.DialogUtility;
import static cl.ui.fx.FactoryUtility.isEnclosingClassEqual;
import cl.ui.fx.OBJSettingDialogFX;
import static cl.ui.fx.TreeCellMaterialSourceFX.MATERIAL_FORMAT;
import coordinate.parser.obj.OBJInfo;
import filesystem.core.OutputInterface;
import java.io.File;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.event.ActionEvent;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.stage.FileChooser;
import jfx.dialog.DialogFXMain;
import jfx.dialog.types.ProcessDialog;

/**
 * FXML Controller class
 *
 * @author user
 */
public class UserInterfaceFXMLController implements Initializable, OutputInterface, RenderControllerInterface<TracerAPI, MaterialFX> {
    @FXML
    StackPane viewportPane;
    @FXML
    StackPane displayPane;
    @FXML
    BorderPane mainPane;
    @FXML
    Button render;
    @FXML
    TreeView<MaterialFX> source;
    @FXML
    TreeView<MaterialFX> destination;
    
    @FXML
    Slider fovSlider;
    @FXML
    Label fovLabel;
    @FXML
    Button fovResetButton;
    @FXML
    Button sceneboundButton;
    
    //Environment
    @FXML
    Button loadenvButton;
    @FXML
    ImageView envmapImageView;
    @FXML
    CheckBox setenvmapCheckBox;
    
    /**
     * Initializes the controller class.
     */
    
    private TracerAPI api;         
    private final OrientationModel<CPoint3, CVector3, CRay, CBound> orientation = new OrientationModel(CPoint3.class, CVector3.class);
    private int currentInstance = -2;
    private BitmapRGBE bitmapRGBE = null;
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {  
        
        TreeItem<MaterialFX> rootsource = new TreeItem<>(new MaterialFX("Root Source"));
        rootsource.getChildren().add(new TreeItem(new MaterialFX("Diffuse 1")));
        rootsource.getChildren().add(new TreeItem(new MaterialFX("Diffuse 2")));
        rootsource.getChildren().add(new TreeItem(new MaterialFX("Diffuse 3")));
        rootsource.getChildren().add(new TreeItem(new MaterialFX("Diffuse 4")));
        source.setRoot(rootsource);        
        rootsource.setExpanded(true);
        
        TreeItem rootdestination = new TreeItem<>(new MaterialFX("Root Destination"));
        destination.setRoot(rootdestination);
        
        //set tree call back interface for custom render and events
        source.setCellFactory(new TreeCellMaterialSourceFX());
        destination.setCellFactory(new TreeCellMaterialDestinationFX());
        
        //render, pause, stop and back to rt      
        render.setOnAction(e -> {
            api.setDevicePriority(RENDER);
            api.getDeviceGI().start();   
            DialogUtility.showAndWait(mainPane, new RenderDialog(api));            
        });
        
        //field of view
        fovLabel.textProperty().bind(
                Bindings.format("%.1f degrees", fovSlider.valueProperty())
        );        
        fovSlider.valueProperty().addListener((observable, oldValue, newValue) -> {
                api.getDeviceRT().getCameraModel().fov = newValue.floatValue();
                api.getDeviceRT().resume();
        });
        fovResetButton.setOnAction(e -> {
            fovSlider.valueProperty().setValue(45);
        });
        sceneboundButton.setOnAction(e -> {
            CDeviceRT device = api.getDeviceRT();
            CBound bound = device.getBound();
            device.setPriorityBound(bound);
            
            device.reposition(bound);
            device.resume();
        });
        
        loadenvButton.setOnAction(e->{
            Optional<Path> path = DialogUtility.showAndWait(UtilityHandler.getScene(), UtilityHandler.getGallery("environment"));
            HDRBitmapReader reader = new HDRBitmapReader();
            
            if(path.isPresent())
            {
                bitmapRGBE = reader.load(path.get());
                envmapImageView.setImage(bitmapRGBE.getScaledImage(500, 500, 2.2));
                api.setEnvironmentMap(bitmapRGBE.getFloat4Data(), bitmapRGBE.getWidth(), bitmapRGBE.getHeight());
            }
        });
        
        setenvmapCheckBox.disableProperty().bind(envmapImageView.imageProperty().isNull());
        setenvmapCheckBox.setOnAction(e->{
            if(setenvmapCheckBox.isSelected())
                api.setIsEnvmapPresent(true);
            else
                api.setIsEnvmapPresent(false);
            api.getDeviceRT().resume();
        });
        
    }    
    
    
    
    @Override
    public void displaySceneMaterial(ArrayList<MaterialT> materials) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setAPI(TracerAPI api) {
        //set api
        this.api = api;
        
        //setup display
        setupDisplay(api);
    }

    private void setupDisplay(TracerAPI api)
    {
        api.getBlendDisplayRT().translationDepth.addListener((observable, old_value, new_value) -> {  
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            orientation.translateDistance(api.getDeviceRT().getCameraModel(), 
                    new_value.floatValue() * api.getDeviceRT().getPriorityBound().getMaximumExtent());     
            api.getDeviceRT().resume();
        });
        
        api.getBlendDisplayRT().translationXY.addListener((observable, old_value, new_value) -> {   
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            orientation.rotateX(api.getDeviceRT().getCameraModel(), (float) new_value.getX());
            orientation.rotateY(api.getDeviceRT().getCameraModel(), (float) new_value.getY());
            api.getDeviceRT().resume();            
        });
        
        //enter drag component
        api.getBlendDisplayRT().setOnDragOver(e -> {
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;   
            
            Point2D xy = api.getBlendDisplayRT().getDragOverXY(e, RAYTRACE_IMAGE.name());
            
            //precisely for material 
            if(isEnclosingClassEqual(e.getGestureSource(), "TreeCellMaterialDestinationFX"))            
            {
                if(api.getDeviceRT().isCoordinateAnInstance(xy.getX(), xy.getY()))
                    e.acceptTransferModes(TransferMode.COPY_OR_MOVE);                 
            }
            
            if(!e.isAccepted()) 
            {
                if(true)
                {
                    BitmapARGB selectionBitmap = api.getDeviceRT().getOverlay().getNull();
                    api.getBlendDisplayRT().set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);            
                    currentInstance = -2;
                    return;
                }
            }
            
            //get instance in current pixel
            int instance = api.getDeviceRT().getInstanceValue(xy.getX(), xy.getY());
            
            //since if we paint in every mouse movement, 
            //it will be expensive in a slow processor, 
            //hence we avoid such a situation.
            //It would still work if we neglet such a concern!!
            if(currentInstance != instance) 
            {
                currentInstance = instance;
                BitmapARGB selectionBitmap = api.getDeviceRT().getOverlay().getDragOverlay(instance);
                api.getBlendDisplayRT().set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);                
            }
        });
        
        //exit drag
        api.getBlendDisplayRT().setOnDragExited(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            BitmapARGB selectionBitmap = api.getDeviceRT().getOverlay().getNull();
            api.getBlendDisplayRT().set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);            
            currentInstance = -2;
        });
        
        //drop material in
        api.getBlendDisplayRT().setOnDragDropped(e -> {
            
            if(!api.isDevicePriority(RAYTRACE)) return;
           
            if(isEnclosingClassEqual(e.getGestureSource(), "TreeCellMaterialDestinationFX"))
            {                               
                MaterialFX matFX = (MaterialFX) e.getDragboard().getContent(MATERIAL_FORMAT);
                matFX.param1.texture.set(UtilityHandler.getAndRemoveImageDnD());      
                api.setMaterial(currentInstance, matFX);
                api.getDeviceRT().resume();
            }
        });
        
        //add display component
        viewportPane.getChildren().add(api.getBlendDisplayRT());
    }

    @Override
    public void print(String key, String string) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
    public void showOBJStatistics(OBJInfo info)
    {
        DialogUtility.showAndWaitFX(UtilityHandler.getScene(), new OBJSettingDialogFX(info));        
    }
    
    public void openOBJFile(ActionEvent e)
    {
        FileChooser chooser = new FileChooser();      
        File file = chooser.showOpenDialog(UtilityHandler.getScene().getWindow());
        if(file != null)
        {
            ProcessDialog dialog = new ProcessDialog();
            DialogUtility.showAndWaitThread(UtilityHandler.getScene(), dialog, (type)->{
                api.initMesh(file.toURI());
                api.getDeviceRT().resume();
                return true;
            });
        }
    }
    
}
