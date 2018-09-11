/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.viewmodel;

import cl.core.device.RayDeviceMesh;
import cl.ui.mvc.model.CustomData;
import static cl.ui.mvc.model.CustomData.Type.PARENT;
import coordinate.parser.attribute.MaterialT;
import java.util.ArrayList;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
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
    private static TreeItem<CustomData> sceneRoot = null;    
    private static TreeItem<CustomData> sceneMaterialTreeItem = null;
    private static TreeItem<CustomData> sceneGroupTreeItem = null;
    
    private static TreeItem<CustomData<MaterialT>> materialRoot = null;    
    private static TreeItem<CustomData<MaterialT>> diffuseTreeItem = null;
    private static TreeItem<CustomData<MaterialT>> emitterTreeItem = null;
    
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
        sceneRoot = new TreeItem(new CustomData("Scene", null));
        sceneMaterialTreeItem = new TreeItem<>(new CustomData("Material", null, PARENT));     
        sceneGroupTreeItem = new TreeItem<>(new CustomData("Group", null, PARENT));
        treeView.setRoot(sceneRoot);
        
        sceneRoot.getChildren().add(sceneGroupTreeItem);
        sceneRoot.getChildren().add(sceneMaterialTreeItem);        
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
                
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Blue" , new MaterialT(0, 0, 1))));
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Red"  , new MaterialT(1, 0, 0))));
        diffuseTreeItem.getChildren().add(new TreeItem(new CustomData<>("Green", new MaterialT(0, 1, 0))));
        
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("10 kW", new MaterialT(0, 0, 0, 0.9f, 0.9f, 0.9f))));
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("20 kW", new MaterialT(0, 0, 0, 0.9f, 0.9f, 0.9f))));
        emitterTreeItem.getChildren().add(new TreeItem(new CustomData<>("30 kW", new MaterialT(0, 0, 0, 0.9f, 0.9f, 0.9f))));
        
        materialRoot.setExpanded(true);
        diffuseTreeItem.setExpanded(true);
        emitterTreeItem.setExpanded(true);
    }
    
    public static void clearSceneMaterial()
    {
        sceneMaterialTreeItem.getChildren().clear();
    }
    
    public static void addSceneMaterial(CustomData material)
    {
        sceneMaterialTreeItem.getChildren().add(new TreeItem(material));
        sceneMaterialTreeItem.setExpanded(true);
    }
                  
    public static void setSceneMaterial(ArrayList<MaterialT> materials)
    {
        clearSceneMaterial();
        materials.forEach((mat) -> sceneMaterialTreeItem.getChildren().add(new TreeItem<>(new CustomData(mat.name, mat))));
        sceneMaterialTreeItem.setExpanded(true);
    }
}
