/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.main;

import bitmap.display.ImageDisplay;
import cl.struct.CRay;
import cl.struct.CBound;
import bitmap.image.BitmapARGB;
import bitmap.image.BitmapRGBE;
import bitmap.reader.HDRBitmapReader;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.abstracts.MambaAPIInterface.DeviceType.RENDER;
import cl.abstracts.MambaAPIInterface.ImageType;
import static cl.abstracts.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import static cl.abstracts.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.abstracts.RenderControllerInterface;
import cl.data.CPoint3;
import cl.data.CVector3;
import cl.fx.UtilityHandler;
import cl.ui.fx.BlendDisplay;
import cl.ui.fx.FactoryUtility;
import cl.ui.fx.render.RenderDialog;
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
import cl.ui.fx.OBJSettingDialogFX;
import cl.ui.fx.SingleTaskFX;
import cl.ui.fx.TreeCellMaterialDestinationFX2;
import static cl.ui.fx.TreeCellMaterialDestinationFX2.MATERIAL_DEST;
import cl.ui.fx.TreeCellMaterialSourceFX2;
import cl.ui.fx.material.MaterialFX2;
import coordinate.parser.obj.OBJInfo;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIcon;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import filesystem.core.OutputInterface;
import filesystem.core.file.FileObject;
import java.nio.file.Path;
import java.util.Optional;
import javafx.beans.binding.Bindings;
import javafx.event.ActionEvent;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.image.ImageView;
import static javafx.scene.input.MouseButton.PRIMARY;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import jfx.IntegerStringConverter;
import jfx.dialog.type.DialogInformation;
import jfx.dialog.type.DialogProcess;

/**
 * FXML Controller class
 *
 * @author user
 */
public class UserInterfaceFXMLController implements Initializable, OutputInterface, RenderControllerInterface<TracerAPI, MaterialFX2> {
    @FXML
    StackPane rootPane;
    
    @FXML
    StackPane viewportPane;
    @FXML
    StackPane displayPane;
    @FXML
    BorderPane mainPane;
    @FXML
    Button render;
    @FXML
    TreeView<MaterialFX2> source;
    @FXML
    TreeView<MaterialFX2> destination;
    
    @FXML
    Slider fovSlider;
    @FXML
    Label fovLabel;
    @FXML
    Button fovResetButton;
    @FXML
    Button sceneboundButton;
    @FXML
    ProgressIndicator renderPortApplyIndicator;
    @FXML
    Spinner<Integer> renderPortWidth;
    @FXML
    Spinner<Integer> renderPortHeight;
    
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
    
    SingleTaskFX taskFX = new SingleTaskFX();
    
    RenderDialog renderDialog = null;
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {  
        
        TreeItem<MaterialFX2> rootsource = new TreeItem<>(new MaterialFX2("Material"));      
        rootsource.getChildren().addAll(getMaterials().getChildren());
        
       
        source.setRoot(rootsource);        
        rootsource.setExpanded(true);
        
        
        
        TreeItem rootdestination = new TreeItem<>(new MaterialFX2("Root Destination"));
        destination.setRoot(rootdestination);
        
        //set tree call back interface for custom render and events
        source.setCellFactory(new TreeCellMaterialSourceFX2());
        destination.setCellFactory(new TreeCellMaterialDestinationFX2());
        
        //render, pause, stop and back to rt      
        render.setOnAction(e -> {
            api.setDevicePriority(RENDER);
            api.getDeviceGI().start();   
            api.getDisplay(ImageDisplay.class).reset();
            
            if(renderDialog != null)
                rootPane.getChildren().remove(renderDialog);
            renderDialog = new RenderDialog(rootPane, api);        
            rootPane.getChildren().add(renderDialog);
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
            api.repositionCameraToSceneRT();
            api.getDeviceRT().resume();
        });
        
        loadenvButton.setOnAction(e->{
            
            Optional<Path> path = FactoryUtility.getHDRGallery().showAndWait(UtilityHandler.getScene());
            HDRBitmapReader reader = new HDRBitmapReader();
            
            if(path.isPresent())
            {
                bitmapRGBE = reader.load(path.get());
                envmapImageView.setImage(bitmapRGBE.getScaledImage(500, 500, 2.2));
                api.setEnvironmentMap(bitmapRGBE);
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
        
        renderPortApplyIndicator.setOpacity(0);   
        //renderPortWidth
        SpinnerValueFactory factoryRW = new SpinnerValueFactory.IntegerSpinnerValueFactory(200, 2500, 500);        
        renderPortWidth.setValueFactory(factoryRW);  
        IntegerStringConverter.createFor(renderPortWidth);
        
        
        //renderPortHeight     
        SpinnerValueFactory factoryRH = new SpinnerValueFactory.IntegerSpinnerValueFactory(200, 2500, 500);
        renderPortHeight.setValueFactory(factoryRH);  
        IntegerStringConverter.createFor(renderPortHeight);
        
        
    }    
    
    private TreeItem<MaterialFX2> getMaterials()
    {
        TreeItem<MaterialFX2> materials = new TreeItem(new MaterialFX2("Glossy"));
        
        MaterialFX2 diffuseWhite = new MaterialFX2("diffuse_white");        
        diffuseWhite.setDiffuseColor(0.9f, 0.9f, 0.9f);   
        diffuseWhite.setDiffuseAmount(1);
        materials.getChildren().add(new TreeItem<>(diffuseWhite));
        
        MaterialFX2 diffuseKhaki = new MaterialFX2("diffuse_khaki");   
        diffuseKhaki.setDiffuseAmount(1);
        diffuseKhaki.setDiffuseColor(0.7647f, 0.6902f, 0.5686f);        
        materials.getChildren().add(new TreeItem<>(diffuseKhaki));
        
        MaterialFX2 diffuseRed = new MaterialFX2("diffuse_red");    
        diffuseRed.setDiffuseAmount(1);
        diffuseRed.setDiffuseColor(0.9f, 0.125f, 0.125f);        
        materials.getChildren().add(new TreeItem<>(diffuseRed));
        
        MaterialFX2 diffuseGreen = new MaterialFX2("diffuse_green");     
        diffuseGreen.setDiffuseAmount(1);
        diffuseGreen.setDiffuseColor(0.125f, 0.9f, 0.125f);        
        materials.getChildren().add(new TreeItem<>(diffuseGreen));
        
        MaterialFX2 steel = new MaterialFX2("glossy_steel");
        steel.setDiffuseAmount(0);
        steel.setGlossyAmount(1);
        steel.setGlossRoughness(0.124f);
        steel.setGlossyColor(0.6f, 0.6f, 0.6f);        
        materials.getChildren().add(new TreeItem<>(steel));
        
        MaterialFX2 gold = new MaterialFX2("glossy_gold");
        gold.setDiffuseAmount(0);
        gold.setGlossyAmount(1);
        gold.setGlossRoughness(0.124f);
        gold.setGlossyColor(0.9f, 0.6f, 0.3f);        
        materials.getChildren().add(new TreeItem<>(gold));
        
        return materials;
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
        api.getDisplay(BlendDisplay.class).translationDepth.addListener((observable, old_value, new_value) -> {  
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            orientation.translateDistance(api.getDeviceRT().getCameraModel(), 
                    new_value.floatValue() * api.getDeviceRT().getPriorityBound().getMaximumExtent());     
            api.getDeviceRT().resume();
        });
        
        api.getDisplay(BlendDisplay.class).translationXY.addListener((observable, old_value, new_value) -> {   
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            orientation.rotateX(api.getDeviceRT().getCameraModel(), (float) new_value.getX());
            orientation.rotateY(api.getDeviceRT().getCameraModel(), (float) new_value.getY());
            api.getDeviceRT().resume();            
        });
        
        //enter drag component
        api.getDisplay(BlendDisplay.class).setOnDragOver(e -> {
            //current device that is rendering is raytrace
            if(!api.isDevicePriority(RAYTRACE)) return;   
            
            Point2D xy = api.getDisplay(BlendDisplay.class).getDragOverXY(e, RAYTRACE_IMAGE.name());
            
            //precisely for material 
            if(e.getDragboard().hasContent(MATERIAL_DEST))            
            {
                
                if(api.getDeviceRT().isCoordinateAnInstance(xy.getX(), xy.getY()))
                {
                    e.acceptTransferModes(TransferMode.COPY_OR_MOVE);
                    
                }                 
            }
            
            if(!e.isAccepted()) 
            {
                
                if(true)
                {
                    BitmapARGB selectionBitmap = api.getDeviceRT().getOverlay().getNull();
                    api.getDisplay(BlendDisplay.class).set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);            
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
                api.getDisplay(BlendDisplay.class).set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);                
            }
        });
        
        //exit drag
        api.getDisplay(BlendDisplay.class).setOnDragExited(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            BitmapARGB selectionBitmap = api.getDeviceRT().getOverlay().getNull();
            api.getDisplay(BlendDisplay.class).set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);            
            currentInstance = -2;
        });
        
        //drop material in
        api.getDisplay(BlendDisplay.class).setOnDragDropped(e -> {
            
            if(!api.isDevicePriority(RAYTRACE)) return;
           
            if(e.getDragboard().hasContent(MATERIAL_DEST))
            {                               
                MaterialFX2 matFX = (MaterialFX2) e.getDragboard().getContent(MATERIAL_DEST);
            
                api.setMaterial(currentInstance, matFX);
                api.getDeviceRT().resume();
            }
        });
        
        api.getDisplay(BlendDisplay.class).setOnMouseClicked(e->{
            if(e.getClickCount() == 2 && e.getButton() == PRIMARY)
            {
                Point2D xy = api.getDisplay(BlendDisplay.class).getMouseOverXY(e, RAYTRACE_IMAGE.name());
                
                //get instance in current pixel
                int instance = api.getDeviceRT().getInstanceValue(xy.getX(), xy.getY());
                
                if(instance > -1)
                {
                    CBound bound = new CBound();
                    api.getDeviceRT().findBound(instance, bound);
                    api.repositionCameraToBoundRT(bound);
                    api.getDeviceRT().resume();
                }
            }
        });
        
        //add display component
        viewportPane.getChildren().add(api.getDisplay(BlendDisplay.class));
    }

    @Override
    public void print(String key, String string) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
    public boolean showOBJStatistics(OBJInfo info)
    {
        OBJSettingDialogFX objSettingDialog = new OBJSettingDialogFX(info);
        Optional<Boolean> optional = objSettingDialog.showAndWait(UtilityHandler.getScene()); 
        return optional.get();
    }
    
    public void openOBJFile(ActionEvent e)
    {        
        Optional<FileObject> fileOption = FactoryUtility.getOBJFileChooser().showAndWait(
                UtilityHandler.getScene().getWindow());
        if(fileOption.isPresent())
        {
            DialogProcess processDialog = new DialogProcess(300, 100);
            processDialog.setRunnable(()->{
                api.initMesh(fileOption.get().getFile().toURI());
                api.getDeviceRT().resume();
            });
            processDialog.showAndWait(UtilityHandler.getScene());
        }
    }
    
    public void showInformation(ActionEvent e)
    {
        FontAwesomeIconView icon = new FontAwesomeIconView(FontAwesomeIcon.INFO_CIRCLE);
        icon.setSize("48");
        icon.setFill(Color.BLUE);
        DialogInformation dialog = new DialogInformation(""
                + "This is a simple java opencl ray tracer with monte-carlo path tracing"
                + " which is designed to be intuitive as much as possible and user friendly."
                + " Aim is to provide access of state of the art ray tracing with current GPU hardware in your computer."
                + "\n\n"
                + "Currently targeting OpenCL 1.2!", 300, 400);
             
        dialog.showAndWait(UtilityHandler.getScene());
    }
    
    public void applyRenderPortSize(ActionEvent e)
    {
        taskFX.execute(renderPortApplyIndicator, ()->{
            api.setImageSize(RENDER_IMAGE, renderPortWidth.getValue(), renderPortHeight.getValue());
            api.getDeviceGI().setup(renderPortWidth.getValue(), renderPortHeight.getValue());
            System.out.println(renderPortWidth.getValue()+ " " +renderPortHeight.getValue());
        });
    }
    
}
