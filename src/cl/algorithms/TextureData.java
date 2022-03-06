/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import java.util.concurrent.atomic.AtomicInteger;
import javafx.scene.image.Image;

/**
 *
 * @author user
 */
public final class TextureData {
    
    AtomicInteger globalIndex = new AtomicInteger(0);       
    
    public Int4 diffuseTexture;         //x, y-coord, argb, has_texture (0 or 1, false or true)      
    public Int4 glossyTexture;
    public Int4 roughnessTexture;      
    public Int4 mirrorTexture;
    public Int4 parameters;             //x = materialID;
        
    public TextureData()
    {
        diffuseTexture = new Int4(null, 0, globalIndex);        
        glossyTexture = new Int4(null, 1, globalIndex);        
        roughnessTexture = new Int4(null, 2, globalIndex);        
        mirrorTexture = new Int4(null, 3, globalIndex);
        parameters = new Int4(null, 4, globalIndex);
        
    }
    
    public TextureData(int[] array)
    {
        this();
        setArray(array);
        
    }
    
    public boolean hasDiffuseTexture()
    {
        return diffuseTexture.getW() > 0;
    }
    
    public boolean hasGlossyTexture()
    {
        return glossyTexture.getW() > 0;
    }
    
    public boolean hasRoughnessTexture()
    {
        return roughnessTexture.getW() > 0;
    }
    
    public boolean hasMirrorTexture()
    {
        return mirrorTexture.getW() > 0;
    }
    
    public int getMaterialIndex()
    {
        return parameters.getX();
    }
    
    public void setDiffuseArgb(Image image, int ui, int vi)
    {
        diffuseTexture.setZ(image.getPixelReader().getArgb(ui, vi)); //argb value
    }
    
    public void setGlossyArgb(Image image, int ui, int vi)
    {
        glossyTexture.setZ(image.getPixelReader().getArgb(ui, vi)); //argb value
    }
    
    public void setRoughnessArgb(Image image, int ui, int vi)
    {
        roughnessTexture.setZ(image.getPixelReader().getArgb(ui, vi)); //argb value
    }
    
    public void setMirrorArgb(Image image, int ui, int vi)
    {
        mirrorTexture.setZ(image.getPixelReader().getArgb(ui, vi)); //argb value
    }
    
    public float getDiffuseTextureU()
    {        
        return getDomainValue(Float.intBitsToFloat(diffuseTexture.getX())); //u
    }
    
    public float getDiffuseTextureV()
    {
        return getDomainValue(Float.intBitsToFloat(diffuseTexture.getY())); //v
    }
    
    public float getGlossyTextureU()
    {        
        return getDomainValue(Float.intBitsToFloat(glossyTexture.getX())); //u
    }
    
    public float getGlossyTextureV()
    {
        return getDomainValue(Float.intBitsToFloat(glossyTexture.getY())); //v
    }
    
    public float getRoughnessTextureU()
    {        
        return getDomainValue(Float.intBitsToFloat(roughnessTexture.getX())); //u
    }
    
    public float getRoughnessTextureV()
    {
        return getDomainValue(Float.intBitsToFloat(roughnessTexture.getY())); //v
    }
    
    public float getMirrorTextureU()
    {        
        return getDomainValue(Float.intBitsToFloat(mirrorTexture.getX())); //u
    }
    
    public float getMirrorTextureV()
    {
        return getDomainValue(Float.intBitsToFloat(mirrorTexture.getY())); //v
    }
    
    private float getDomainValue(float x)
    {
        //from sunflow Texture.java in method getPixel(float x, float y) 
        //in short, this handles texture that has defined uv coordinates in mesh
        return x < 0 ? x - (int) x + 1 : x - (int) x;
    }
    
    public void setArray(int[] array)
    {
        diffuseTexture.set(array);
        glossyTexture.set(array);
        roughnessTexture.set(array);
        mirrorTexture.set(array);
        parameters.set(array);
    }
    
    public void setIndex(int index)
    {
        this.globalIndex.set(index * intSize());
    }
    
    public int intSize()
    {
        return diffuseTexture.intSize() + 
               glossyTexture.intSize() + 
               roughnessTexture.intSize() +
               mirrorTexture.intSize() + 
               parameters.intSize();
    }
    
    public static class Int4
    {
        int[] array;
        int fieldIndex;
        AtomicInteger globalIndex;
        
        public Int4(int[] array, int fieldIndex, AtomicInteger globalIndex)
        {
            this.array = array;
            this.fieldIndex = fieldIndex;
            this.globalIndex = globalIndex;
        }
        
        public void init(int x, int y, int z, int w)
        {
            setX(x);
            setY(y);
            setZ(z);
            setW(w);
        }
        
        public int getX()
        {
            return array[globalIndex.get() + fieldIndex * intSize() + 0];
        }
        
        public void setX(int value)
        {
            array[globalIndex.get() + fieldIndex * intSize() + 0] = value;
        }
        
        public int getY()
        {
            return array[globalIndex.get() + fieldIndex * intSize() + 1];
        }
        
        public void setY(int value)
        {
            array[globalIndex.get() + fieldIndex * intSize() + 1] = value;
        }
        
        public int getZ()
        {
            return array[globalIndex.get() + fieldIndex * intSize() + 2];
        }
        
        public void setZ(int value)
        {
            array[globalIndex.get() + fieldIndex * intSize() + 2] = value;
        }
        
        public int getW()
        {
            return array[globalIndex.get() + fieldIndex * intSize() + 3];
        }
        
        public void setW(int value)
        {
            array[globalIndex.get() + fieldIndex * intSize() + 3] = value;
        }
        
        public void set(int[] array)
        {
            this.array = array;
        }

        public int intSize()
        {
            return 4; 
        }
    }
}
