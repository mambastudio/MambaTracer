/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.fx;

import java.util.HashMap;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.scene.Scene;
import javafx.scene.image.Image;

/**
 *
 * @author user
 */
public class UtilityHandler {
    private static Scene scene = null;
    private static final ObjectProperty<Image> imageDragAndDrop = new SimpleObjectProperty(null);
    private static final HashMap<String, GalleryDialogFX> galleries = new HashMap();
    
    public static boolean hasDraggedImage()
    {
        return imageDragAndDrop.get() != null;
    }
    
    public static void setImageDnD(Image image)
    {
        imageDragAndDrop.set(image);
    }
    
    public static Image getAndRemoveImageDnD()
    {
        Image image = imageDragAndDrop.get();
        imageDragAndDrop.set(null);
        return image;
    }
    
    public static void setScene(Scene scene)
    {
        UtilityHandler.scene = scene;
    }
    
    public static Scene getScene()
    {
        return scene;
    }
    
    public static void register(String name, GalleryDialogFX dialog)
    {
        galleries.put(name, dialog);
    }
    
    public static GalleryDialogFX getGallery(String name)
    {
        if(!galleries.containsKey(name))
        {
            galleries.put(name, new GalleryDialogFX("Gallery"));
        }
        return galleries.get(name);
    }
    
    //https://stackoverflow.com/questions/4051202/get-the-outer-class-object-from-an-inner-class-object
    public static boolean isEnclosingClassEqual(Object object, String name)
    {
        return object.getClass().getEnclosingClass().getSimpleName().equals(name);
    }
}
