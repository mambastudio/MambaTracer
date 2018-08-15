/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import cl.renderer.SimpleRender;
import bitmap.display.StaticDisplay;
import cl.ui.icons.IconAssetManager;
import cl.ui.mvc.DataModel;
import cl.ui.mvc.CustomData;
import cl.ui.tree.view.MaterialVaultTreeCell;
import cl.ui.tree.view.TargetTreeCell;
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
import javafx.scene.control.Button;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.TextArea;
import javafx.scene.control.TreeView;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.StackPane;
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
        
    private final StaticDisplay display = new StaticDisplay();
    private final SimpleRender render = new SimpleRender();
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO     
        
        //Init material database and scene tree view   
        DataModel.initMaterialTreeData(treeViewMaterial);
        DataModel.initSceneTreeData(treeViewScene);
        
        //Set cell renderer for tree cell
        treeViewScene.setCellFactory(m -> new TargetTreeCell());        
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
                DataModel.addSceneMaterial(data);                
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
        DataModel.clearSceneMaterial();
    }
}
