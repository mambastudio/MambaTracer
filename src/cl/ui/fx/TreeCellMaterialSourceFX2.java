/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.fx.UtilityHandler;
import cl.ui.fx.material.MaterialFX2;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.DataFormat;
import javafx.scene.input.Dragboard;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.TransferMode;
import jfx.tree.TreeCellFactory;

/**
 *
 * @author user
 */
public class TreeCellMaterialSourceFX2 implements TreeCellFactory<MaterialFX2>{
    public static final DataFormat MATERIAL_SOURCE = new DataFormat("MATERIAL_SOURCE");
    
    @Override
    public void dragDetected(MouseEvent event, TreeCell<MaterialFX2> treeCell, TreeView<MaterialFX2> treeView) {
        TreeItem<MaterialFX2> draggedItem = treeCell.getTreeItem();

        //item in cell is null
        if(draggedItem == null) return;

        // root can't be dragged
        if (!draggedItem.isLeaf()) return;
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
