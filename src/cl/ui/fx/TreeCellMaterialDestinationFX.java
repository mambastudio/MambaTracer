/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.ui.fx.material.MaterialFX;
import cl.ui.fx.material.MaterialEditor;
import cl.fx.UtilityHandler;
import java.util.Optional;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.TransferMode;
import javafx.util.Callback;
import jfx.dialog.DialogUtility;
import static cl.ui.fx.FactoryUtility.isEnclosingClassEqual;
import static cl.ui.fx.TreeCellMaterialSourceFX.MATERIAL_FORMAT;

/**
 *
 * @author user
 */
public class TreeCellMaterialDestinationFX implements Callback<TreeView<MaterialFX>, TreeCell<MaterialFX>>{

    @Override
    public TreeCell<MaterialFX> call(TreeView<MaterialFX> treeView) {        
        TreeCell<MaterialFX> cell = new TreeCell<MaterialFX>() {
            @Override
            protected void updateItem(MaterialFX item, boolean empty) {
                super.updateItem(item, empty);
                
                textProperty().unbind();
                
                if(empty || item == null)
                {
                    setGraphic(null);
                    setText(null);            
                }        
                else
                {
                    setText(item.name.get());
                    textProperty().bind(item.name);
                }
            }
        };
        cell.setOnDragOver((DragEvent event) -> dragOver(event, cell, treeView));
        cell.setOnDragDropped((DragEvent event) -> drop(event, cell, treeView));
        cell.setOnDragDetected((MouseEvent event) -> dragDetected(event, cell, treeView));
        
        cell.setOnMouseClicked(e->{
            if(cell.getTreeItem() == null || cell.getTreeItem().getParent() == null)
                return;
            if(e.getClickCount() == 2)
            { 
                if(cell.getItem() != null)
                {    
                    MaterialFX defMat = cell.getItem();
                   
                    Optional<MaterialFX> option = DialogUtility.showAndWait(UtilityHandler.getScene(), new MaterialEditor(defMat, UtilityHandler.getGallery("texture")));
                    if(option.isPresent())
                        defMat.setMaterial(option.get());
                    treeView.requestFocus(); //request focus because if dialog is closed, it loses focus
                   
                }
                
            }
        });
        
        return cell;
    }

    private void dragOver(DragEvent event, TreeCell<MaterialFX> treeCell, TreeView<MaterialFX> treeView) {
        /* data is dragged over the target */
        /* accept it only if it is not dragged from the same node 
         * and if it has a string data */
        if(event.getDragboard().hasContent(MATERIAL_FORMAT))
        {                  
            if(isEnclosingClassEqual(event.getGestureSource(), "TreeCellMaterialSourceFX"))
                event.acceptTransferModes(TransferMode.COPY_OR_MOVE);   
            
        }
        
        event.consume();
        
    }
    
    private void drop(DragEvent event, TreeCell<MaterialFX> treeCell, TreeView<MaterialFX> treeView) {
        /* data dropped */
        /* if there is a string data on dragboard, read it and use it */
        Dragboard db = event.getDragboard();
        boolean success = false;
        if(db.hasContent(MATERIAL_FORMAT))
        {
            MaterialFX mat = (MaterialFX) db.getContent(MATERIAL_FORMAT);
            
            mat.param1.texture.set(UtilityHandler.getAndRemoveImageDnD());
            
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
        
        treeView.refresh();
    }
    
    
    
    private void dragDetected(MouseEvent event, TreeCell<MaterialFX> treeCell, TreeView<MaterialFX> treeView) {
        TreeItem<MaterialFX> draggedItem = treeCell.getTreeItem();
        
        if(draggedItem == null) return;
        
        MaterialFX matFX = draggedItem.getValue();

        // root can't be dragged
        if (draggedItem.getParent() == null) return;
        Dragboard db = treeCell.startDragAndDrop(TransferMode.ANY);
        
        UtilityHandler.setImageDnD(matFX.param1.texture.get());
        

        ClipboardContent content = new ClipboardContent();
        content.put(MATERIAL_FORMAT, draggedItem.getValue());
        db.setContent(content);
        db.setDragView(treeCell.snapshot(null, null));
        event.consume();
    }
}
