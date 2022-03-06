/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.data.CPoint4;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.SimpleFloatProperty;

/**
 *
 * @author user
 */
public class Point4FX  {
    public FloatProperty x;
    public FloatProperty y;
    public FloatProperty z;
    public FloatProperty w;
    
    public Point4FX()
    {
        x = new SimpleFloatProperty();
        y = new SimpleFloatProperty();
        z = new SimpleFloatProperty();
        w = new SimpleFloatProperty();
    }
    
    public Point4FX(CPoint4 p)
    {
        this();
        x.setValue(p.get(0));
        y.setValue(p.get(1));
        z.setValue(p.get(2));
        w.setValue(p.get(3));
    }
    
    public Point4FX(float fx, float fy, float fz, float fw)
    {
        this();
        x.setValue(fx);
        y.setValue(fy);
        z.setValue(fz);
        w.setValue(fw);
    }
    
    private void init()
    {
        x = new SimpleFloatProperty();
        y = new SimpleFloatProperty();
        z = new SimpleFloatProperty();
        w = new SimpleFloatProperty();
    }
    
    public float getX()
    {
        return x.floatValue();
    }
    
    public float getY()
    {
        return y.floatValue();
    }
    
    public float getZ()
    {
        return z.floatValue();
    }
    
    public float getW()
    {
        return w.floatValue();
    }
    
    public void setX(float v)
    {
        x.setValue(v);
    }
    
    public void setY(float v)
    {
        y.setValue(v);
    }
    
    public void setZ(float v)
    {
        z.setValue(v);
    }
    
    public void setW(float v)
    {
        w.setValue(v);
    }
    
    public void set(Point4FX point)
    {
        this.x.set(point.getX());
        this.y.set(point.getY());
        this.z.set(point.getZ());
        this.w.set(point.getW());
    }
    
    public FloatProperty getXProperty()
    {
        return x;
    }
    
    public FloatProperty getYProperty()
    {
        return y;
    }
    
    public FloatProperty getZProperty()
    {
        return z;
    }
    public FloatProperty getWProperty()
    {
        return w;
    }
    
    
    public CPoint4 getCPoint4()
    {
        return new CPoint4(x.floatValue(), y.floatValue(), z.floatValue(), w.floatValue());
    }
    
}
