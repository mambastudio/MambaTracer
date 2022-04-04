/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package editor;

import cl.fx.UtilityHandler;
import static cl.fx.UtilityHandler.isEnclosingClassEqual;
import cl.ui.fx.material.MaterialFX2;
import cl.ui.fx.material.MaterialFX2EditorDialog;
import java.util.Optional;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Scene;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.DataFormat;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import jfx.tree.TreeCellFactory;

/**
 *
 * @author user
 */
public class Test3 extends Application{
    public static final DataFormat MATERIAL_SOURCE = new DataFormat("MATERIAL_SOURCE");
    public static final DataFormat MATERIAL_EDITED = new DataFormat("MATERIAL_EDITED");
    
    public static void main(String... args)
    {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        VBox root = new VBox();
        
        TreeView<MaterialFX2> view1 = new TreeView();
        TreeView<MaterialFX2> view2 = new TreeView();   
        
        TreeItem<MaterialFX2> troot = new TreeItem(new MaterialFX2("Material"));
        TreeItem<MaterialFX2> diffuse = new TreeItem(new MaterialFX2("Diffuse"));
        TreeItem<MaterialFX2> diffuse1 = new TreeItem(new MaterialFX2("Diffuse 1"));
        TreeItem<MaterialFX2> diffuse2 = new TreeItem(new MaterialFX2("Diffuse 2"));
        TreeItem<MaterialFX2> diffuse3 = new TreeItem(new MaterialFX2("Diffuse 3"));
        diffuse.getChildren().addAll(diffuse1, diffuse2, diffuse3);
        troot.getChildren().add(diffuse);
        diffuse.setExpanded(true);
        
        TreeItem<MaterialFX2> mroot = new TreeItem(new MaterialFX2("Scene Material"));
        view1.setRoot(mroot);
        view1.setShowRoot(false);        
        view1.setCellFactory(new TreeMaterialDestinationCellFactory());
        
        view2.setRoot(troot);
        view2.setShowRoot(false);        
        view2.setCellFactory(new TreeMaterialSourceCellFactory());
        
        
        root.getChildren().addAll(view1, view2);
        
        
        Scene scene = new Scene(root, 800, 650);
        UtilityHandler.setScene(scene);
        
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    
    public static class TreeMaterialSourceCellFactory implements TreeCellFactory<MaterialFX2>
    {
        @Override
        public void dragDetected(MouseEvent event, TreeCell<MaterialFX2> treeCell, TreeView<MaterialFX2> treeView) {
            TreeItem<MaterialFX2> draggedItem = treeCell.getTreeItem();
       
            //item in cell is null
            if(draggedItem == null) return;
        
            MaterialFX2 matFX = draggedItem.getValue();

            // root can't be dragged
            if (draggedItem.getParent() == null) return;
            Dragboard db = treeCell.startDragAndDrop(TransferMode.ANY);
        

            ClipboardContent content = new ClipboardContent();
            content.put(MATERIAL_SOURCE, draggedItem.getValue());
            db.setContent(content);
            db.setDragView(treeCell.snapshot(null, null));
            event.consume();
        }
        
        @Override
        public String getSpecialName()
        {
            return "MaterialSource";
        }
    }
    
    public static class TreeMaterialDestinationCellFactory implements TreeCellFactory<MaterialFX2>
    {
        @Override
        public void dragOver(DragEvent event, TreeCell<MaterialFX2> treeCell, TreeView<MaterialFX2> treeView) {
            /* data is dragged over the target */
            /* accept it only if it is not dragged from the same node 
             * and if it has a string data */
            if(event.getDragboard().hasContent(MATERIAL_SOURCE))
            {                      
                if(isEnclosingClassEqual(event.getGestureSource(), "TreeCellFactory"))
                {                   
                    event.acceptTransferModes(TransferMode.COPY_OR_MOVE);
                }  
            }

            event.consume();

        }
    
        @Override
        public void drop(DragEvent event, TreeCell<MaterialFX2> treeCell, TreeView<MaterialFX2> treeView) {
            /* data dropped */
            /* if there is a string data on dragboard, read it and use it */
            Dragboard db = event.getDragboard();
            boolean success = false;
            if(db.hasContent(MATERIAL_SOURCE))
            {
                MaterialFX2 mat = (MaterialFX2) db.getContent(MATERIAL_SOURCE);

                //get root
                TreeItem item = new TreeItem(mat);
                treeView.getRoot().getChildren().add(item);
                treeView.getRoot().setExpanded(true);

                //scroll to new material and select it
                int index = treeView.getRoot().getChildren().indexOf(item);
                treeView.scrollTo(index+1);
                treeView.getSelectionModel().select(index+1);
                success = true;
            }
           /* let the source know whether the material was successfully 
            * transferred and used */
            event.setDropCompleted(success);
            event.consume();
            db.clear();

            treeView.refresh();
        }
        
        @Override
        public void updateIfPresent(TreeCell<MaterialFX2> cell, TreeItem<MaterialFX2> treeItem, MaterialFX2 item)
        {
            cell.setText(item.name.get());
            cell.textProperty().bind(item.name);
        }        
        
        @Override
        public void mouseCellClicked(TreeCell<MaterialFX2> cell, TreeItem<MaterialFX2> treeItem, MouseEvent e)
        {
            if(treeItem == null || treeItem.getParent() == null)
                return;
            if(e.getClickCount() == 2)
            { 
                MaterialFX2 defMat = cell.getItem();
                MaterialFX2EditorDialog materialDialog = new MaterialFX2EditorDialog(defMat, UtilityHandler.getGallery("texture"));
                Optional<MaterialFX2> option = materialDialog.showAndWait(UtilityHandler.getScene());
                if(option.isPresent())
                    defMat.setMaterial(option.get());
                cell.requestFocus(); //request focus because if dialog is closed, it loses focus
            }
        }
    }
}
