/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.env.sun.model;

import bitmap.Color;
import bitmap.core.AbstractBitmap;
import bitmap.display.DynamicDisplay;
import bitmap.image.BitmapBGRA;
import cl.core.env.sun.data.ArHosekSkyModelState;
import cl.core.env.sun.data.SphericalCoordinate;
import thread.model.LambdaThread;

/**
 *
 * @author user
 */
public class SunskyRender {
    private DynamicDisplay display;
    
    private ArHosekSkyModelState skyState;
    private ArHosekSkyModelState sunState;
    
    private double elevation = 0;
    private double turbidity = 1;
    private double albedo = 0;
    private double tonemap = 3.2;
    private double exposure = 0.01;
    private double sunsize = 12.9;
    
    private boolean fast;
    
    LambdaThread thread = new LambdaThread();
    
    
    public SunskyRender()
    {
        
    }
    
    public void startExecution(DynamicDisplay display)
    {
        initState();
        
        this.display = display;
        thread.startExecution(() -> {
            render();
        });
    }
    
    public void render()
    {
        AbstractBitmap tile;
        
        if(display != null)
        {
            for(int i = 3; i >= 0; i--)
            {                         
                thread.chill();
            
                int w = display.getImageWidth();
                int h = display.getImageHeight();
                
                //just in case the ui is not initialized and w and h = 1
                if(w < 100) w = 100;
                if(h < 100) h = 100;

                tile = new BitmapBGRA(w, h);
               
                int step = 1 << i;
                if(fast && i != 3) return;
                
                boolean finished = trace(tile, 0, 0, w, h, step, i);

                if(!finished) return;     
                                                
                display.imageFill(tile);                
                fast = false;            
            }
        }   
        
        thread.pauseExecution();
    }
    
    private void initState()
    {
        sunState = HosekWilkie.initStateRadiance(turbidity, albedo, SphericalCoordinate.elevationDegrees((float)elevation));
        skyState = HosekWilkie.initStateRGB(turbidity, albedo, SphericalCoordinate.elevationDegrees((float)elevation));
    }
    
    public boolean trace(AbstractBitmap tile, int startX, int startY, int endX, int endY, int step, int i)
    {
        boolean finished = false;
        int w = tile.getWidth(); int h = tile.getHeight();
        
        for(int y = startY; y < endY; y+=step)
            for(int x = startX; x < endX; x+=step)
            {

                thread.chill();
                if(fast && i != 3) return finished;

                int wi = x + step >= endX ? endX - x : step;
                int hi = y + step >= endY ? endY - y : step;

                Color sun = HosekWilkie.getRGB_using_solar_radiance(SphericalCoordinate.sphericalDirection(x, y, w, h), sunState, exposure, tonemap);
                Color sky = HosekWilkie.getRGB(SphericalCoordinate.sphericalDirection(x, y, w, h), skyState, exposure, tonemap);
                float sunAlpha = Math.min(sun.getMin(), 1);

                Color col = Color.blend(sky, sun, sunAlpha);    

                tile.writeColor(col, 1, x, y, wi, hi);                    
            }
        finished = true;
        return finished;
    }
            
    public void stop() {
        thread.stopExecution();
    }
    
    public void pause() {
        thread.pauseExecution();
    }

    public void resume() {
        fast = true;
        thread.resumeExecution();
    }
    
    public void shutdownThread()
    {
        thread.stopExecution();
    }
}
