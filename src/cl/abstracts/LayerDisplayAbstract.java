/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.abstracts;

import bitmap.core.AbstractDisplay;
import bitmap.core.BitmapInterface;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.geometry.Bounds;
import javafx.geometry.Point2D;
import javafx.scene.image.ImageView;
import javafx.scene.input.DragEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.StackPane;

/**
 *
 * @author user
 */
public abstract class LayerDisplayAbstract extends StackPane implements AbstractDisplay
{
    //Adding listeners to these triggers mouse dragging events
    public DoubleProperty translationDepth = new SimpleDoubleProperty();
    public ObjectProperty<Point2D> translationXY = new SimpleObjectProperty<>();
      
    //Image view array
    protected HashMap<String, ImageView> imageArray;
    
    //Mouse location
    protected double mouseLocX, mouseLocY; 
    
    public LayerDisplayAbstract()
    {
        this.imageArray = new HashMap<>();     
        this.setOnMousePressed(this::mousePressed);
        this.setOnMouseDragged(this::mouseDragged);    
    } 
    
    public abstract void initDisplay(String... layers);       
    public abstract void mousePressed(MouseEvent e);
    public abstract void mouseDragged(MouseEvent e);
    
    public ImageView get(String name)
    {
        return imageArray.get(name);
    }
    
    public void set(String name, BitmapInterface bitmap)
    {
        boolean namePresent = false;
        List<String> list = new ArrayList<>(imageArray.keySet());
        namePresent = list.stream().map((string) -> string.equals(name)).reduce(namePresent, (accumulator, _item) -> accumulator | _item);       
        if(namePresent)
        {         
            Platform.runLater(() ->{
                this.get(name).setImage(bitmap.getImage());
            });        
        }
        
    }
    
    public int getImageWidth(String name) {        
        return (int) get(name).imageProperty().get().getWidth();
    }

    
    public int getImageHeight(String name) {
        return (int) get(name).imageProperty().get().getWidth();
    }           
    
    @Override
    public void imageFill(String name, BitmapInterface bitmap) {
        Platform.setImplicitExit(false);
        Platform.runLater(() -> {      
            this.get(name).setImage(bitmap.getImage());
        });
        
    } 
    
    public Bounds getScreenBounds(String name)
    {
        return get(name).localToScreen(get(name).getBoundsInLocal());
    }
    
    public Point2D getDragOverXY(DragEvent e, String name)
    {
        Bounds screenBound = getScreenBounds(name);
        double x = e.getScreenX() - screenBound.getMinX();
        double y = e.getScreenY() - screenBound.getMinY();
        return new Point2D(x, y);
    }    
    
    public Point2D getMouseOverXY(MouseEvent e, String name)
    {
        Bounds screenBound = getScreenBounds(name);
        double x = e.getScreenX() - screenBound.getMinX();
        double y = e.getScreenY() - screenBound.getMinY();
        return new Point2D(x, y);
    }
}
