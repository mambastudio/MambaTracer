/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.view;

import cl.ui.mvc.view.icons.IconAssetManager;
import cl.ui.mvc.model.CustomData;
import coordinate.parser.attribute.GroupT;
import coordinate.parser.attribute.MaterialT;
import filesystem.fx.icons.FileIconManager;
import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.beans.binding.Bindings;
import javafx.beans.value.ChangeListener;
import javafx.scene.control.TreeCell;
import javafx.scene.control.TreeItem;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;

/**
 *
 * @author user
 */
public class TargetTreeCell extends TreeCell<CustomData>{
    //Apparently, in a treeview, only one tree cell is used for the whole tree, 
    //hence care must be taken in which the tree item data doesn't register the
    //same listener more than once when added from tree cell updateItem()
    InvalidationListener listener;
    
    public TargetTreeCell()
    {
        setOnDragDetected(e ->{            
            Dragboard db = startDragAndDrop(TransferMode.COPY);
            ClipboardContent content = new ClipboardContent();        
            TargetTreeCell cell = (TargetTreeCell)e.getSource();
            
            if(!cell.getTreeItem().isLeaf()) return;
            
            CustomData data = cell.getItem();
            
            if(!(data.getData() instanceof MaterialT)) return;
            
            content.put(CustomData.getFormat(), data);
            db.setContent(content);
            e.consume();
        });
        
        listener = o -> {
            TreeItem<CustomData> item = this.getTreeItem();
            
            //The item might be null sometimes
            if(item!= null)
            {
                CustomData data = item.getValue();
                setGraphic(IconAssetManager.getIcon((MaterialT)data.getData()));
            }            
        };
        
    }

    @Override
    public void updateItem(CustomData customData, boolean empty)
    {
        super.updateItem(customData, empty);
        
        textProperty().unbind();
        
        
        if(empty || customData == null)
        {
            setGraphic(null);
            setText(null);            
        }        
        else
        {                      
            if(getTreeItem().getParent() == null)
                setGraphic(FileIconManager.getIcon("home"));            
            else if(customData.getData() instanceof MaterialT)                  
                setGraphic(IconAssetManager.getIcon((MaterialT)customData.getData()));
            else if(customData.getData() instanceof GroupT)                  
                setGraphic(IconAssetManager.getGroupIcon());
            setText(customData.getName());
            
            textProperty().bind(customData.getNameProperty());
            
            //remove and add same listener to prevent duplicates
            customData.getQProperty().removeListener(listener);
            customData.getQProperty().addListener(listener);
        
        }           
    }
}
