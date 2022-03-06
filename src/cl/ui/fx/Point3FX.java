/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import cl.data.CPoint3;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.SimpleFloatProperty;

/**
 *
 * @author user
 */
public class Point3FX {
    public FloatProperty x;
    public FloatProperty y;
    public FloatProperty z;
    
    public Point3FX()
    {
        init();
    }
    
    public Point3FX(CPoint3 p)
    {
        this();
        x.setValue(p.get(0));
        y.setValue(p.get(1));
        z.setValue(p.get(2));
    }
    
    public Point3FX(float fx, float fy, float fz)
    {
        this();
        x.setValue(fx);
        y.setValue(fy);
        z.setValue(fz);
    }
    
    private void init()
    {
        x = new SimpleFloatProperty();
        y = new SimpleFloatProperty();
        z = new SimpleFloatProperty();
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
    
    public void set(Point3FX point)
    {
        this.x.set(point.getX());
        this.y.set(point.getY());
        this.z.set(point.getZ());
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
    
    public CPoint3 getCPoint3()
    {
        return new CPoint3(x.floatValue(), y.floatValue(), z.floatValue());
    }
    
    public void set(CPoint3 p)
    {
        x.setValue(p.get(0));
        y.setValue(p.get(1));
        z.setValue(p.get(2));
    }
    
}
