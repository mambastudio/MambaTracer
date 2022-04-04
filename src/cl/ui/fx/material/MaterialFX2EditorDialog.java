/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.fx.GalleryDialogFX;
import java.util.Arrays;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import jfx.dialog.DialogContent;
import jfx.dialog.DialogExtend;

/**
 *
 * @author user
 */
public class MaterialFX2EditorDialog extends DialogExtend<MaterialFX2>  {
    MaterialFX2Editor materialEditor;
    GalleryDialogFX dialog;
    
    public MaterialFX2EditorDialog(MaterialFX2 defMat, GalleryDialogFX dialogImages)
    {
        this.materialEditor = new MaterialFX2Editor(defMat.copy());
        this.dialog = dialogImages;
        setup();        
    }
    
    @Override
    public void setup() {
        //dialog content
        DialogContent<Boolean> settingContent = new DialogContent<>();
        settingContent.setContent(materialEditor, DialogContent.DialogStructure.HEADER_FOOTER);
        
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
                return materialEditor.getEditedMaterial();
            else
                return null;
        });  
    }
}
