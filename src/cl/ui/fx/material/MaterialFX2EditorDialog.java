/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material;

import cl.fx.GalleryDialogFX;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import jfx.dialog.DialogAbstract;

/**
 *
 * @author user
 */
public class MaterialFX2EditorDialog extends DialogAbstract<MaterialFX2>  {
    MaterialFX2Editor materialEditor;
    GalleryDialogFX dialog;
    
    public MaterialFX2EditorDialog(MaterialFX2 defMat, GalleryDialogFX dialogImages)
    {
        this.materialEditor = new MaterialFX2Editor(defMat.copy());
        this.dialog = dialogImages;
        
        init();
    }
    
    public final void init()
    {
        this.setButtons(OK, CANCEL);    
        this.setContent(materialEditor);
        this.setSize(700, 550);
        
        this.setSupplier((buttonType)->{
            if(buttonType.equals(OK))
                return materialEditor.getEditedMaterial();
            else
                return null;
        });
    }
}
