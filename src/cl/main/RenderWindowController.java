/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import cl.core.console.Console;
import bitmap.image.BitmapARGB;
import cl.core.CBoundingBox;
import cl.core.api.MambaAPIInterface;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.core.api.MambaAPIInterface.DeviceType.RENDER;
import cl.core.api.MambaAPIInterface.ImageType;
import static cl.core.api.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import static cl.core.api.MambaAPIInterface.ImageType.RENDER_IMAGE;
import cl.core.api.RayDeviceInterface;
import cl.core.api.RenderControllerInterface;
import cl.core.data.CPoint3;
import cl.core.data.CVector3;
import cl.core.data.struct.CRay;
import cl.core.device.RayDeviceMesh;
import cl.ui.mvc.view.icons.IconAssetManager;
import cl.ui.mvc.viewmodel.RenderViewModel;
import cl.ui.mvc.model.CustomData;
import cl.ui.mvc.view.MaterialVaultTreeCell;
import cl.ui.mvc.view.TargetTreeCell;
import com.sun.javafx.scene.control.skin.LabeledText;
import coordinate.model.OrientationModel;
import coordinate.parser.attribute.MaterialT;
import filesystem.core.OutputFactory;
import filesystem.util.FileChooserManager;
import filesystem.util.FileUtility;
import filesystem.util.FileUtility.FileOption;
import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.ResourceBundle;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Bounds;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ColorPicker;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseButton;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.scene.shape.Circle;
import javafx.stage.FileChooser;
import javafx.stage.Window;
import thread.model.LambdaThread;

/**
 * FXML Controller class
 *
 * @author user
 */
public class RenderWindowController implements Initializable, RenderControllerInterface<TracerAPI, RayDeviceInterface,MaterialT> {

    /**
     * Initializes the controller class.
     */
    
    @FXML
    BorderPane pane;
    @FXML
    StackPane parentPane;
    
    @FXML
    TextArea timeConsole;
    @FXML
    TextArea sceneConsole;
    @FXML
    TextArea performanceConsole;
    
    @FXML
    private TreeView treeViewScene;
    @FXML
    private TreeView treeViewMaterial;
    
    @FXML
    Button renderButton;
    @FXML
    Button pauseButton;
    @FXML
    Button stopButton;
    @FXML 
    Button editButton;
    
    
    @FXML
    Button openButton;
    
    //MaterialEditor
    @FXML
    TextField nameTextField;
    @FXML
    ColorPicker diffuseColorPicker;
    @FXML
    Slider diffuseWeightSlider;
    @FXML
    Label diffuseWeightLabel;
        
    @FXML
    ColorPicker reflectionColorPicker;
    @FXML
    Spinner<Double> exponentialU;
    @FXML
    Spinner<Double> exponentialV;
    @FXML
    Spinner<Double> iorSpinner;
    
    @FXML
    ColorPicker emitterColorPicker;
    @FXML
    Spinner<Double> emitterPowerSpinner;
    
    @FXML
    GridPane diffuseGridPane;
    @FXML
    GridPane reflectionGridPane;
    @FXML
    GridPane refractionGridPane;
    
    @FXML
    CheckBox refractionEnabled;
    @FXML
    CheckBox emitterEnabled;
    
    
    //Tab section
    @FXML
    TabPane tPane;
    @FXML
    Tab brdfTab;
    @FXML
    Tab matsTab;
        
    //API and display     
    private MambaAPIInterface api;
         
    //Javafx data
    private TreeItem<CustomData<MaterialT>> sceneRoot = null;    
    
    //Javafx data for material & group     
    private TreeItem<CustomData<MaterialT>> materialRoot = null;    
    private TreeItem<CustomData<MaterialT>> diffuseTreeItem = null;
    private TreeItem<CustomData<MaterialT>> emitterTreeItem = null;
    
    OrientationModel<CPoint3, CVector3, CRay, CBoundingBox> orientation = new OrientationModel(CPoint3.class, CVector3.class);
    int currentinstance = -2;
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
                
        // TODO    
        sceneRoot = new TreeItem(new CustomData("Scene Material", null)); 
        sceneRoot.setExpanded(true);
       
        //Init material database and scene tree view   
        initMaterialTreeData(treeViewMaterial);
        initSceneTreeData(treeViewScene);
        
        //Set cell renderer for tree cell       
        treeViewScene.setCellFactory(m -> new TargetTreeCell());                 
        treeViewScene.setOnMouseClicked(e -> { //this selection is for cell material leaf only
            if(e.getClickCount() == 2)
            {                
                Node node = e.getPickResult().getIntersectedNode();
                if(node instanceof TargetTreeCell)
                {
                    TargetTreeCell tCell = (TargetTreeCell)node;
                    if(tCell.getTreeItem() != null)
                    {
                        TreeItem item = tCell.getTreeItem(); 
                        if(item.isLeaf())
                        {
                            CustomData data = (CustomData) item.getValue();
                            MaterialT mat = (MaterialT) data.getData();
                            RenderViewModel.materialEditorModel.initMaterial(mat);
                            RenderViewModel.cmat = data;
                            tPane.getSelectionModel().select(brdfTab);
                        }
                    }
                }
                else if(node instanceof Circle || node instanceof LabeledText)
                {
                    
                    TreeItem item = (TreeItem) treeViewScene.getSelectionModel().getSelectedItem();
                    if(item.isLeaf())
                    {
                        CustomData data = (CustomData) item.getValue();
                        MaterialT mat = (MaterialT) data.getData();
                        RenderViewModel.materialEditorModel.initMaterial(mat);
                        RenderViewModel.cmat = data;
                        tPane.getSelectionModel().select(brdfTab);
                    }
                }
                //else
                //    System.out.println(node);
            }
        });
        treeViewMaterial.setCellFactory(m -> new MaterialVaultTreeCell());       
        initDragAndDrop();
        
        //Set Console
        Console console = new Console(timeConsole, sceneConsole, performanceConsole);        
        OutputFactory.setOutput(console);     
        
        //Set FileChooser
        FileChooserManager.init("objscene");
        FileChooser objsceneChooser = FileChooserManager.getFileChooser("objscene");        
        objsceneChooser.getExtensionFilters().setAll(new FileChooser.ExtensionFilter("Wavefront obj", "*.obj"));
        
        
        //Read init directory from file if exists or create file
        if(FileUtility.fileExistsInTemporaryDirectory("ctracer.txt"))
        {
            Path dataFile = FileUtility.getFilePathFromTemporaryDirectory("ctracer.txt");
            FileUtility.readLines(dataFile, line -> {
                if(line.contains("scenepath"))                 
                    objsceneChooser.setInitialDirectory(new File(line.split("=")[1].trim()));               
                else
                    objsceneChooser.setInitialDirectory(new File("."));
            });
        }
        else
        {
            FileUtility.createFile(Paths.get("ctracer.txt"), FileOption.IN_TEMPORARY_DIR_DONT_DELETE);
            objsceneChooser.setInitialDirectory(new File("."));            
        }
        
        //Bind directory property in case of change of directory
        FileChooserManager.getBindingFileProperty("objscene").addListener((observable, oldValue, newValue) -> {
            Path dataFile = FileUtility.getFilePathFromTemporaryDirectory("ctracer.txt");
            FileUtility.writeLines(dataFile, "scenepath = " +newValue);            
        });
        
        //icons for buttons (file management)
        openButton.setGraphic(IconAssetManager.getOpenIcon());
        
        //icons for buttons (dealing with rendering)
        renderButton.setGraphic(IconAssetManager.getRenderIcon());
        pauseButton.setGraphic(IconAssetManager.getPauseIcon());
        stopButton.setGraphic(IconAssetManager.getStopIcon());
        editButton.setGraphic(IconAssetManager.getEditIcon());        
        
        //button states
        pauseButton.setDisable(true);
        stopButton.setDisable(true); 
        editButton.setDisable(true);
        
        //button listeners
        renderButton.setOnAction(e -> {

            pauseButton.setDisable(false);
            stopButton.setDisable(false);
            renderButton.setDisable(true);
            editButton.setDisable(true);
            
            api.setDevicePriority(RENDER);            
            
            if(api.getDevice(RENDER).isStopped())           
                api.startDevice(RENDER);                  
            else if(api.getDevice(RENDER).isPaused())           
                api.resumeDevice(RENDER); 
        });
        pauseButton.setOnAction(e -> {            
            pauseButton.setDisable(true);
            stopButton.setDisable(false);
            renderButton.setDisable(false);
            editButton.setDisable(true);
            
            api.pauseDevice(RENDER);
        });
        stopButton.setOnAction(e -> {
            pauseButton.setDisable(true);
            stopButton.setDisable(true);
            renderButton.setDisable(false);
            editButton.setDisable(false);
            
            api.stopDevice(RENDER);
        });
        editButton.setOnAction(e -> {            
            api.applyImage(RENDER_IMAGE, () -> {
                return new BitmapARGB(api.getImageSize(RENDER_IMAGE).x, api.getImageSize(RENDER_IMAGE).y, false);
            });
            editButton.setDisable(true);
            
            api.setDevicePriority(RAYTRACE);
        });
        
        //Editor register
        RenderViewModel.materialEditorModel.registerNameTextField(nameTextField);
        
        RenderViewModel.materialEditorModel.registerDiffuseColorPicker(diffuseColorPicker);
        RenderViewModel.materialEditorModel.registerDiffuseWeightSlider(diffuseWeightSlider);
        RenderViewModel.materialEditorModel.registerDiffuseWeightLabel(diffuseWeightLabel);
         
        RenderViewModel.materialEditorModel.registerReflectionColorPicker(reflectionColorPicker);
        RenderViewModel.materialEditorModel.registerExponentialUSpinner(exponentialU);
        RenderViewModel.materialEditorModel.registerExponentialVSpinner(exponentialV);
        RenderViewModel.materialEditorModel.registerRefractionEnabled(refractionEnabled);
        RenderViewModel.materialEditorModel.registerIORSpinner(iorSpinner);
        
        RenderViewModel.materialEditorModel.registerEmitterColorPicker(emitterColorPicker);
        RenderViewModel.materialEditorModel.registerEmitterPowerSpinner(emitterPowerSpinner);
        RenderViewModel.materialEditorModel.registerEmitterEnabled(emitterEnabled);
        
        RenderViewModel.materialEditorModel.initMaterial(new MaterialT());
        
        //gui to gui component interaction
        diffuseGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        reflectionGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        refractionGridPane.disableProperty().bind(emitterEnabled.selectedProperty());
        emitterColorPicker.disableProperty().bind(emitterEnabled.selectedProperty().not());
        emitterPowerSpinner.disableProperty().bind(emitterEnabled.selectedProperty().not());
        
        iorSpinner.disableProperty().bind(refractionEnabled.selectedProperty().not());
      
    } 
    
    public void initDragAndDrop()
    {
        treeViewScene.setOnDragOver(e ->{
            if(!(e.getGestureSource() instanceof TargetTreeCell))            
                if(e.getDragboard().getContent(CustomData.getFormat()) instanceof CustomData)
                {                    
                    e.acceptTransferModes(TransferMode.COPY_OR_MOVE);
                }
            
            e.consume();
        });
        
        treeViewScene.setOnDragDropped(e -> {
            /* data dropped */
            /* if there is a string data on dragboard, read it and use it */
            Dragboard db = e.getDragboard();
            boolean success = false;
            if(db.hasContent(CustomData.getFormat()))
            {
                CustomData data = (CustomData) e.getDragboard().getContent(CustomData.getFormat());
                addSceneMaterial(data);                
                success = true;
            }
             /* let the source know whether the string was successfully 
              * transferred and used */
              e.setDropCompleted(success);
              e.consume();
        });
    }
    
    public void exit(ActionEvent e)
    {
        System.exit(0);
    }
    
   
    public void open(ActionEvent e)
    {
        api.pauseDevice(RAYTRACE);
        
        File file = launchSceneFileChooser(null);        
        if(file == null) {api.resumeDevice(RAYTRACE); return;}
        
        LambdaThread.executeThread(()->{
            ProgressIndicator progressIndicator = new ProgressIndicator();
            Platform.runLater(() -> parentPane.getChildren().add(progressIndicator));
            api.initMesh(file.toPath());
            api.resumeDevice(RAYTRACE);
            Platform.runLater(() -> parentPane.getChildren().remove(progressIndicator));
        });
    }
    
    public File launchSceneFileChooser(Window window)
    {                
        File file = FileChooserManager.showOpenDialog("objscene");
        return file;
    }
    
    public void resetSceneTreeMaterial(ActionEvent e)
    {
       clearSceneMaterial();
    }
    
    public void backToMaterialsTab(ActionEvent e)
    {
        tPane.getSelectionModel().select(matsTab);
    }
    
    public void acceptEditedMaterial(ActionEvent e)
    {
        RenderViewModel.cmat.setMaterial(RenderViewModel.materialEditorModel.getEditedMaterial());
    }

    @Override
    public void setAPI(TracerAPI api) {
        this.api = api;
        this.pane.setCenter(api.getBlendDisplay());         
        
        api.getBlendDisplay().translationDepth.addListener((observable, old_value, new_value) -> {               
            if(!api.isDevicePriority(RAYTRACE)) return;
            orientation.translateDistance(api.getDevice(RAYTRACE).getCamera(), new_value.floatValue() * api.getDevice(RAYTRACE).getBound().getMaximumExtent());     
            api.getDevice(RAYTRACE).resume();
        });
        
        api.getBlendDisplay().translationXY.addListener((observable, old_value, new_value) -> {    
            if(!api.isDevicePriority(RAYTRACE)) return;
            orientation.rotateX(api.getDevice(RAYTRACE).getCamera(), (float) new_value.getX());
            orientation.rotateY(api.getDevice(RAYTRACE).getCamera(), (float) new_value.getY());
            api.getDevice(RAYTRACE).resume();
        });
        
        api.getBlendDisplay().setOnDragOver(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;            
            Bounds imageViewInScreen = api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).localToScreen(api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).getBoundsInLocal());
            double x = e.getScreenX() - imageViewInScreen.getMinX();
            double y = e.getScreenY() - imageViewInScreen.getMinY();
            
            if(!(e.getGestureSource() instanceof MaterialVaultTreeCell))            
                if(e.getDragboard().getContent(CustomData.getFormat()) instanceof CustomData)
                {         
                    if(api.overlay.isInstance(x, y))
                        e.acceptTransferModes(TransferMode.COPY_OR_MOVE);             
                }
                       
            int instance = api.overlay.get(x, y);
            
            //since if we paint in every mouse movement, 
            //it will be expensive in a slow processor, 
            //hence we avoid such a situation.
            //It would still work if we neglet such a concern!!
            if(currentinstance != instance) 
            {
                currentinstance = instance;
                BitmapARGB selectionBitmap = api.overlay.getDragOverlay(instance);
                api.getBlendDisplay().set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);                
            }
        });
        api.getBlendDisplay().setOnDragExited(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            BitmapARGB selectionBitmap = api.overlay.getNull();
            api.getBlendDisplay().set(ImageType.OVERLAY_IMAGE.name(), selectionBitmap);            
            currentinstance = -2;
        });
        api.getBlendDisplay().setOnDragDropped(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            Bounds imageViewInScreen = api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).localToScreen(api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).getBoundsInLocal());
            double x = e.getScreenX() - imageViewInScreen.getMinX();
            double y = e.getScreenY() - imageViewInScreen.getMinY();
            
            if(e.getGestureSource() instanceof TargetTreeCell)
            {
                CustomData data = (CustomData) e.getDragboard().getContent(CustomData.getFormat());
                MaterialT mat = (MaterialT) data.getData();       
                int cmatIndex = api.overlay.get(x, y);          
                api.getDevice(RAYTRACE).setMaterial(cmatIndex, mat);
                api.getDevice(RAYTRACE).resume();
            }
        });
        
        api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).setOnMousePressed(e -> {
            if(!api.isDevicePriority(RAYTRACE)) return;
            
            if(e.getButton().equals(MouseButton.PRIMARY)){
                if(e.getClickCount() == 2){
                    
                    Bounds imageViewInScreen = api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).localToScreen(api.getBlendDisplay().get(RAYTRACE_IMAGE.name()).getBoundsInLocal());
                    double x = e.getScreenX() - imageViewInScreen.getMinX();
                    double y = e.getScreenY() - imageViewInScreen.getMinY();
                    
                    int instance = api.overlay.get(x, y);
                    //System.out.println(instance);
                    if(instance > -1)
                    {
                        RayDeviceMesh device = (RayDeviceMesh)api.getDevice(RAYTRACE);
                        CBoundingBox bound = device.getGroupBound(instance);
                        device.reposition(bound);
                        device.resume();
                    }
                }               
            }
            
        });
    }
    
    public void initSceneTreeData(TreeView treeView)
    {
        sceneRoot = new TreeItem(new CustomData("Scene Material", null));      
        treeView.setRoot(sceneRoot);               
        sceneRoot.setExpanded(true);
    }
    
    public void initMaterialTreeData(TreeView treeView)
    {
        diffuseTreeItem = new TreeItem(new CustomData("Diffuse", null));
        emitterTreeItem = new TreeItem(new CustomData("Emitter", null));
        
        materialRoot = new TreeItem(new CustomData("Material Vault", null)); 
        treeView.setRoot(materialRoot);
        
        materialRoot.getChildren().add(diffuseTreeItem); 
        materialRoot.getChildren().add(emitterTreeItem);
                
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Blue" , new MaterialT("Blue", 0, 0, 1))));
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Red"  , new MaterialT("Red", 1, 0, 0))));
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Green", new MaterialT("Green", 0, 1, 0))));
        
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("10 kW", new MaterialT("10 kW", 0, 0, 0, 0.9f, 0.9f, 0.9f))));
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("20 kW", new MaterialT("20 kW", 0, 0, 0, 0.9f, 0.9f, 0.9f))));
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("30 kW", new MaterialT("30 kW", 0, 0, 0, 0.9f, 0.9f, 0.9f))));
        
        materialRoot.setExpanded(true);
        diffuseTreeItem.setExpanded(true);
        emitterTreeItem.setExpanded(true);
    }
    
    public void clearSceneMaterial()
    {
        sceneRoot.getChildren().clear();
    }
    
    public void addSceneMaterial(CustomData material)
    {
        sceneRoot.getChildren().add(new TreeItem(material));
        sceneRoot.setExpanded(true);
    }
         
    @Override
    public void displaySceneMaterial(ArrayList<MaterialT> materials)
    {       
        Platform.runLater(() -> {
            clearSceneMaterial();
            materials.forEach((mat) -> sceneRoot.getChildren().add(new TreeItem<>(new CustomData(mat.name, mat))));
            sceneRoot.setExpanded(true);
        });        
        
    }             

}
