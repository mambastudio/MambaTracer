/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import cl.renderer.SimpleRender;
import bitmap.display.StaticDisplay;
import cl.ui.mvc.view.icons.IconAssetManager;
import cl.ui.mvc.viewmodel.RenderViewModel;
import cl.ui.mvc.model.CustomData;
import cl.ui.mvc.view.MaterialVaultTreeCell;
import cl.ui.mvc.view.TargetTreeCell;
import com.sun.javafx.scene.control.skin.LabeledText;
import coordinate.parser.attribute.MaterialT;
import filesystem.core.OutputFactory;
import filesystem.util.FileChooserManager;
import filesystem.util.FileUtility;
import filesystem.util.FileUtility.FileOption;
import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ResourceBundle;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
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
public class RenderWindowController implements Initializable {

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
        
    private final StaticDisplay display = new StaticDisplay();
    private final SimpleRender render = new SimpleRender();
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO     
        
        //Init material database and scene tree view   
        RenderViewModel.initMaterialTreeData(treeViewMaterial);
        RenderViewModel.initSceneTreeData(treeViewScene);
        
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
        
        //Init rest of gui
        pane.setCenter(display);        
        render.launch(display);
        render.close();        
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
                RenderViewModel.addSceneMaterial(data);                
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
        render.pauseKernel();
        
        File file = launchSceneFileChooser(null);
        
        if(file == null) {render.resumeKernel(); return;}
        
        LambdaThread.executeThread(()->{
            ProgressIndicator progressIndicator = new ProgressIndicator();
            Platform.runLater(() -> parentPane.getChildren().add(progressIndicator));
            render.initMesh(file.toPath());            
            render.resumeKernel();
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
        RenderViewModel.clearSceneMaterial();
    }
    
    public void backToMaterialsTab(ActionEvent e)
    {
        tPane.getSelectionModel().select(matsTab);
    }
    
    public void acceptEditedMaterial(ActionEvent e)
    {
        RenderViewModel.cmat.setMaterial(RenderViewModel.materialEditorModel.getEditedMaterial());
    }
}
