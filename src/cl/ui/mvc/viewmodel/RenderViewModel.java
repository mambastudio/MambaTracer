/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.viewmodel;

import bitmap.display.BlendDisplay;
import bitmap.image.BitmapARGB;
import cl.core.CMaterialInterface;
import cl.core.Overlay;
import cl.core.device.RayDeviceMesh;
import cl.ui.mvc.model.CustomData;
import coordinate.parser.attribute.MaterialT;
import java.util.ArrayList;
import javafx.application.Platform;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;

/**
 *
 * @author user
 */
public class RenderViewModel 
{
    //OpenCL mesh (gateway to opencl)
    private static RayDeviceMesh device;
    
    //Javafx data
    private static TreeItem<CustomData<MaterialT>> sceneRoot = null;    
    
    //Javafx data for material & group     
    private static TreeItem<CustomData<MaterialT>> materialRoot = null;    
    private static TreeItem<CustomData<MaterialT>> diffuseTreeItem = null;
    private static TreeItem<CustomData<MaterialT>> emitterTreeItem = null;
    
    //MaterialEditorModel
    public static MaterialEditorModel materialEditorModel = new MaterialEditorModel();
    public static CMaterialInterface cmat = null;
    
    public static Overlay overlay = null;
    
    public static boolean isRendering = false;
    
    public static BlendDisplay display = null;
    public static BitmapARGB renderBitmap = null;
    
    public static void setDevice(RayDeviceMesh device)
    {
        RenderViewModel.device = device;
    }
    
    public static RayDeviceMesh getDevice()
    {
        return device;
    }
    
    public static void initSceneTreeData(TreeView treeView)
    {
        sceneRoot = new TreeItem(new CustomData("Scene Material", null));      
        treeView.setRoot(sceneRoot);               
        sceneRoot.setExpanded(true);
    }
    
    public static void initMaterialTreeData(TreeView treeView)
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
    
    public static void clearSceneMaterial()
    {
        sceneRoot.getChildren().clear();
    }
    
    public static void addSceneMaterial(CustomData material)
    {
        sceneRoot.getChildren().add(new TreeItem(material));
        sceneRoot.setExpanded(true);
    }
                  
    public static void setSceneMaterial(ArrayList<MaterialT> materials)
    {
       
        Platform.runLater(() -> {
             clearSceneMaterial();
            materials.forEach((mat) -> sceneRoot.getChildren().add(new TreeItem<>(new CustomData(mat.name, mat))));
            sceneRoot.setExpanded(true);
        });        
        
    }       
}
