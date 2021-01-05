/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.ui.fx.material.MaterialFX;
import cl.fx.UtilityHandler;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.DataFormat;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.TransferMode;
import javafx.util.Callback;

/**
 *
 * @author user
 */
public class TreeCellMaterialSourceFX implements Callback<TreeView<MaterialFX>, TreeCell<MaterialFX>>{
    public static final DataFormat MATERIAL_FORMAT = new DataFormat("MATERIAL");
    
    @Override
    public TreeCell<MaterialFX> call(TreeView<MaterialFX> treeView) {
        TreeCell<MaterialFX> cell = new TreeCell<MaterialFX>() {
            @Override
            protected void updateItem(MaterialFX item, boolean empty) {
                super.updateItem(item, empty);
                if(empty || item == null)
                {
                    setGraphic(null);
                    setText(null);            
                }        
                else
                    setText(item.name.get());
            }
        };
        cell.setOnDragDetected((MouseEvent event) -> dragDetected(event, cell, treeView));
        
        return cell;
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
