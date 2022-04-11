/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import javafx.geometry.Point2D;
import javafx.scene.effect.BlendMode;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import cl.abstracts.LayerDisplayAbstract;

/**
 *
 * @author user
 */
public class BlendDisplay extends LayerDisplayAbstract {
   
    public BlendDisplay(String... layers)
    {
        super();
        initDisplay(layers);
    }
    
    @Override
    public final void initDisplay(String... layers)
    {
        for(int i = 0; i<layers.length; i++)
            if(i == 0)
                imageArray.put(layers[i], new ImageView());
            else
            {
                imageArray.put(layers[i], new ImageView());
                imageArray.get(layers[i]).setBlendMode(BlendMode.SRC_OVER);
            }
        
        for(String name : layers)        
            getChildren().add(imageArray.get(name));                    
    }
        
    @Override
    public void mousePressed(MouseEvent e)
    {     
        mouseLocX = e.getX();
        mouseLocY = e.getY();        
    }
    
    @Override
    public void mouseDragged(MouseEvent e)
    {        
        float currentLocX = (float) e.getX();
        float currentLocY = (float) e.getY();
        
        float dx = (float) (currentLocX - mouseLocX);
        float dy = (float) (currentLocY - mouseLocY);
      
        if (e.isSecondaryButtonDown())
        {
            translationDepth.setValue(dy * 0.1f);            
        }
        else
        {
            Point2D pointxy = new Point2D(-dx*0.5, -dy*0.5);
            translationXY.setValue(pointxy);    
            
        }
        this.mouseLocX = currentLocX;
        this.mouseLocY = currentLocY;                    
    }   

    @Override
    public int getImageWidth() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getImageHeight() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
}
