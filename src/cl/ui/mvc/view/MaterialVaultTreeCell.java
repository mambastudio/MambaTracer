/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.view;

import cl.ui.mvc.view.icons.IconAssetManager;
import cl.ui.mvc.model.CustomData;
import coordinate.parser.attribute.MaterialT;
import filesystem.fx.icons.FileIconManager;
import javafx.scene.control.TreeCell;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;

/**
 *
 * @author user
 * 
 * http://jaypthakkar.blogspot.com/2013/11/javafx-updating-item-in-tableview.html
 * 
 */
public class MaterialVaultTreeCell  extends TreeCell<CustomData<MaterialT>>{
    public MaterialVaultTreeCell()
    {       
        
        setOnDragDetected(e ->{            
            Dragboard db = startDragAndDrop(TransferMode.COPY);
            ClipboardContent content = new ClipboardContent();        
            MaterialVaultTreeCell cell = (MaterialVaultTreeCell)e.getSource();
           
            if(!cell.getTreeItem().isLeaf()) return;
            
            CustomData data = cell.getItem();
            content.put(CustomData.getFormat(), data);
            db.setContent(content);
            e.consume();
        });
                
    }
    @Override
    public void updateItem(CustomData<MaterialT> item, boolean empty)
    {
        super.updateItem(item, empty);
        if(empty)
        {
            setGraphic(null);
            setText(null);
            
        }
        
        else
        {
            
            if(getTreeItem().getParent() == null)
                setGraphic(FileIconManager.getIcon("home"));
            else if(getTreeItem().getChildren().size() > 0)
                setGraphic(FileIconManager.getIcon("folder"));
            else
                setGraphic(IconAssetManager.getIcon(item.getData()));
            setText(item.getName());
        }            
            
    }
}
