/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data;

import coordinate.generic.AbstractCoordinateFloat;
import coordinate.struct.FloatStruct;
import javafx.scene.paint.Color;

/**
 *
 * @author user
 */
public class CColor4 extends FloatStruct  implements AbstractCoordinateFloat {
    public float r, g, b, w;
    
    public CColor4()
    {
        r = g = b = 0;
        w = 1;
    }
    
    public CColor4(float r, float g, float b)
    {
        this.r = r; this.g = g; this.b = b;
        this.w = 1;
    }
    
    public CColor4(float r, float g, float b, float w)
    {
        this(r, g, b);
        this.w = w;
    }
    @Override
    public int getSize() {
        return 4;
    }

    @Override
    public float[] getArray() {
        return new float[]{r, g, b, w};
    }

    @Override
    public float get(char axis) {
        switch (axis) {
            case 'r':
                return r;
            case 'g':
                return g;
            case 'b':
                return b;
            case 'w':
                return w;
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.  
        }
    }

    @Override
    public void set(char axis, float value) {
        switch (axis) {
            case 'r':
                r = value;
                break;
            case 'g':
                g = value;
                break;
            case 'b':
                b = value;
                break;
            case 'w':
                w = value;
                break;
            default:
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
        this.refreshGlobalArray();
    }

    @Override
    public void set(float... values) {        
        for(int i = 0; i<values.length; i++)
            setIndex(i, values[i]);
        this.refreshGlobalArray();
    }
    
    public void set(double... values){
        for(int i = 0; i<values.length; i++)
            setIndex(i, (float) values[i]);
        this.refreshGlobalArray();
    }
    
    public void setColorFX(Color color)
    {
        set(color.getRed(), color.getGreen(), color.getBlue());
    }
    
    public Color getColorFX()
    {
        return new Color(r, g, b, w);
    }

    @Override
    public void setIndex(int index, float value) {
        switch (index)
        {
            case 0:
                r = value;
                break;
            case 1:
                g = value;
                break;    
            case 2:
                b = value;
                break;
            case 3:
                w = value;
                break;    
        }
        this.refreshGlobalArray();
    }        

    @Override
    public int getByteSize() {
        return 4;
    }
    
    public CColor4 copy()
    {
        return new CColor4(r, g, b, w);
    }
    
    @Override
    public String toString()
    {
        float[] array = getArray();
        return String.format("(%3.2f, %3.2f, %3.2f, %3.2f)", array[0], array[1], array[2], array[3]);
    }
}
