/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import bitmap.Color;
import bitmap.image.BitmapARGB;

/**
 *
 * @author user
 */
public class Overlay {
    int w, h;
    int wh;
    int [] instances = null;

    Color selected = Color.GREEN;

    public Overlay(int w, int h)
    {
        this.w = w;
        this.h = h;
        this.wh = w*h;
        instances = new int[wh];
    }
    
    public void setInstanceArray(int[] instances)
    {
        if(instances == null || instances.length != wh) return;        
        this.instances = instances;
    }
    
    public void copyToArray(int[] instances)
    {
        if(instances == null || instances.length != wh) return;
        System.arraycopy(instances, 0, this.instances, 0, wh);
    }
    
    public boolean isInstance(double x, double y)
    {
        return get(x, y) > -1;
    }
    
    public int get(double x, double y)
    {
        return get((int)x, (int) y);
    }
    
    public int get(int x, int y)
    {
        if ((x >= this.w) || (y >= this.h))
            return -2;

        if((x < 0) || (y < 0))
            return -2;
        
        int i = x + y * this.w;
        if (i >= this.wh)
            return -2;

        return instances[i];
    }
    
    public int getPixelIndex(double x, double y)
    {       
        if ((x >= this.w) || (y >= this.h))
            return -1;

        return (int) (x + y * this.w);        
    }

    public void set(int x, int y, int instance)
    {
        if ((x >= this.w) || (y >= this.h))
            return;

        int i = x + y * this.w;
        if (i >= this.wh)
            return;

        this.instances[i] = instance;
    }

    public void set(int x, int y, int w, int h, int instance)
    {
        for (int dx = 0; dx < w; dx++)
            for (int dy = 0; dy < h; dy++)
                set(x + dx, y + dy, instance);
    }

    public synchronized BitmapARGB getDragOverlay(int instance)
    {
        BitmapARGB image = new BitmapARGB(this.w, this.h);

        int i = 0;
        int stride = 6;
        
        /**
         * Borrowed from the discontinued but awesome Radium java monte-carlo renderer but a similar code can be gotten from here
         *         ftp://ftp.ecs.csus.edu/clevengr/155/LectureNotes/Fall12/09ProceduralTexturingLectureNotes.pdf
         */
        for (int y = 0; y < this.h; y++)
        {
            for (int x = 0; x < this.w; x++)
            {
                int mod = (x / stride % 2 + y / stride % 2) % 2;
                if (this.instances[i] == instance)
                {
                    image.writeColor(new Color(mod, mod, mod), 1, x, y);                    
                }
                else
                    image.writeColor(Color.BLACK, 0, x, y);
                i++;
            }
        }
        return image;
    }
    public BitmapARGB getSelectionOverlay(int instance)
    {
        BitmapARGB tile = new BitmapARGB(w, h);
        //tile.setAlphaNull(0);

        int i = -1;

        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                i++;

                if(instance == instances[i])
                {
                    if(x > 0)
                    {
                        int inst = this.instances[(i - 1)];
                        if (inst != instance)
                        {
                            tile.writeColor(selected, 1, x, y);                            
                            continue;
                        }
                    }

                    if (x < this.w - 1)
                    {
                        int inst = this.instances[(i + 1)];
                        if (inst != instance)
                        {
                            tile.writeColor(selected, 1, x, y);                            
                            continue;
                        }
                    }

                    if (y > 0)
                    {
                        int inst = this.instances[(i - this.w)];
                        if ((inst != instance))
                        {
                            tile.writeColor(selected, 1, x, y);
                            continue;
                        }
                    }

                    if (y < this.h - 1)
                    {
                        int inst = this.instances[(i + this.w)];
                        if ((inst != instance))
                        {
                            tile.writeColor(selected, 1, x, y);
                        }
                    }
                }
            }
        return tile;
    }

    public BitmapARGB getNull()
    {
        return new BitmapARGB(this.w, this.h, false);       
    }
}
