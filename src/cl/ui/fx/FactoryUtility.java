/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import static bitmap.display.gallery.GalleryCanvas.ImageType.HDR;
import cl.fx.GalleryDialogFX;
import cl.fx.UtilityHandler;
import filesystem.core.file.FileObject;
import static filesystem.core.file.FileObject.ExploreType.FOLDER;
import filesystem.explorer.FileExplorer;
import java.util.Optional;
import jfx.dialog.DialogUtility;

/**
 *
 * @author user
 */
public class FactoryUtility {
    private static FileChooser objChooser = null;
    private static FileChooser hdrChooser = null;
    private static FileChooser texChooser = null;
    
    private static GalleryDialogFX hdrGallery = null;
    
    //https://stackoverflow.com/questions/4051202/get-the-outer-class-object-from-an-inner-class-object
    public static boolean isEnclosingClassEqual(Object object, String name)
    {        
        return object.getClass().getEnclosingClass().getSimpleName().equals(name);
    }
    
    public static FileChooser getOBJFileChooser()
    {
        if(objChooser == null)
        {
            objChooser = new FileChooser();
            objChooser.addExtensions(new FileExplorer.ExtensionFilter("OBJ", ".obj"));
        }
        return objChooser;
    }
    
    public static FileChooser getHDRFileChooser()
    {
        if(hdrChooser == null)
        {
            hdrChooser = new FileChooser(FOLDER);
            hdrChooser.addExtensions(new FileExplorer.ExtensionFilter("HDR", ".hdr"));
        }
        return hdrChooser;
    }
    
    public static GalleryDialogFX getHDRGallery()
    {
        if(hdrGallery == null)
        {
            hdrGallery = new GalleryDialogFX("environment", HDR);
            hdrGallery.setLaunchDialog(()->{
                Optional<FileObject> fileOption = DialogUtility.showAndWait(UtilityHandler.getScene(), getHDRFileChooser());
                if(fileOption.isPresent())
                {
                    System.out.println(fileOption.get().getPath());
                    hdrGallery.addFolderImages(fileOption.get().getPath());
                }
            });
        }
        return hdrGallery;
    }
}
