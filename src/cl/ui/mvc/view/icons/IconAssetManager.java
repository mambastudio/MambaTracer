/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.view.icons;

import coordinate.parser.attribute.MaterialT;
import filesystem.fx.icons.FileIconManager;
import javafx.scene.Node;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;

/**
 *
 * @author user
 */
public class IconAssetManager {
    public static Node getIcon(MaterialT material)
    {
        if(material == null)
            return FileIconManager.getIcon(IconAssetManager.class, "questionmark20x20.png");
        else if(material.isEmitter())
            return new Circle(10, new Color(material.er, material.eg, material.eb, 1));
        else
            return new Circle(10, new Color(material.dr, material.dg, material.db, 1));
    }
    public static Node getGroupIcon()            
    {
        return FileIconManager.getIcon(IconAssetManager.class, "Fancy20x20.png");
    }
    
    public static Node getOpenIcon()
    {
        return FileIconManager.getIcon(IconAssetManager.class, "Open32x32.png");
    }
    
    public static Node getPauseIcon()
    {
        return FileIconManager.getIcon(IconAssetManager.class, "Pause32x32.png");
    }
    
    public static Node getStopIcon()
    {
        return FileIconManager.getIcon(IconAssetManager.class, "Stop32x32.png");
    }
    
    public static Node getRenderIcon()
    {
        return FileIconManager.getIcon(IconAssetManager.class, "Play32x32.png");
    }
}
