/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.fx;

import bitmap.display.gallery.GalleryCanvas.ImageType;
import bitmap.display.gallery.GalleryLoader;
import bitmap.display.gallery.util.TaskInterface;
import java.nio.file.Path;
import java.util.Arrays;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.layout.BorderPane;
import jfx.dialog.DialogContent;
import jfx.dialog.DialogExtend;

/**
 *
 * @author user
 */
public class GalleryDialogFX extends DialogExtend<Path> {
    
    GalleryLoader loader = new GalleryLoader(400, 400);
    
    public GalleryDialogFX(String message)
    {
        setup();
    }
    public GalleryDialogFX(String message, ImageType... types)
    {
        this(message);
        loader.setImageTypes(types);
    }
    
    public void setLaunchDialog(TaskInterface task)
    {
        loader.setLaunchDialog(task);
    }
    
    public void addFolderImages(Path pathFolder)
    {
        loader.addFolderImages(pathFolder);
    }

    @Override
    public void setup() {
        //dialog content
        DialogContent<Boolean> settingContent = new DialogContent<>();
        settingContent.setContent(loader, DialogContent.DialogStructure.HEADER_FOOTER);
        
        //dialog pane (main window)
        init(
                settingContent,                
                Arrays.asList(
                        new ButtonType("Ok", ButtonBar.ButtonData.OK_DONE),
                        new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE)), 
                700, 550, 
                false);
        
        this.setSupplier((buttonType)->{
            if(buttonType == OK)          
                return loader.getSelectedImagePath();
            else
                return null;
        }); 
    }
}
