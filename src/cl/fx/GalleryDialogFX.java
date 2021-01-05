/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.fx;

import bitmap.display.gallery.GalleryCanvas.ImageType;
import bitmap.display.gallery.GalleryLoader;
import java.nio.file.Path;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.layout.BorderPane;
import jfx.dialog.DialogAbstract;

/**
 *
 * @author user
 */
public class GalleryDialogFX extends DialogAbstract<Path> {
    
    GalleryLoader loader = new GalleryLoader(400, 400);
    
    public GalleryDialogFX(String message)
    {
        
        this.setButtons(OK, CANCEL);
        
        BorderPane pane = new BorderPane();
        pane.setCenter(loader);
        
        this.setContent(pane);
        
        this.setSupplier((buttonType)->{
            if(buttonType == OK)
                return loader.getSelectedImagePath();
            else
                return null;
        });
    }
    public GalleryDialogFX(String message, ImageType... types)
    {
        this(message);
        loader.setImageTypes(types);
    }
}
