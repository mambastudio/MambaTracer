/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.fx.UtilityHandler;
import static cl.fx.UtilityHandler.isEnclosingClassEqual;
import static cl.ui.fx.TreeCellMaterialSourceFX2.MATERIAL_SOURCE;
import cl.ui.fx.material.MaterialFX2;
import cl.ui.fx.material.MaterialFX2EditorDialog;
import java.util.Optional;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.DataFormat;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.TransferMode;
import jfx.tree.TreeCellFactory;

/**
 *
 * @author user
 */
public class TreeCellMaterialDestinationFX2 implements TreeCellFactory<MaterialFX2> 
{
    public static final DataFormat MATERIAL_DEST = new DataFormat("MATERIAL_DESTINATION");
    
    @Override
    public void dragDetected(MouseEvent event, TreeCell<MaterialFX2> treeCell, TreeView<MaterialFX2> treeView) {
        TreeItem<MaterialFX2> draggedItem = treeCell.getTreeItem();

        //item in cell is null
        if(draggedItem == null) return;

       // MaterialFX2 matFX = draggedItem.getValue();

        // root can't be dragged
        if (draggedItem.getParent() == null) return;
        Dragboard db = treeCell.startDragAndDrop(TransferMode.ANY);

        ClipboardContent content = new ClipboardContent();
        content.put(MATERIAL_DEST, draggedItem.getValue());
        db.setContent(content);
        db.setDragView(treeCell.snapshot(null, null));
        event.consume();
    }
    
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

            //mat.param.texture.set(UtilityHandler.getAndRemoveImageDnD());

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
            {                
                defMat.setMaterial(option.get());
            }
            cell.requestFocus(); //request focus because if dialog is closed, it loses focus
        }
    }
    
    @Override
    public String getSpecialName()
    {
        return "MaterialDestination";
    }
}

