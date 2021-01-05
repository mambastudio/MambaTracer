/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.ui.fx.main.TracerAPI;
import cl.struct.CTextureData;
import javafx.scene.image.Image;
import wrapper.core.CMemory;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CTextureApplyPass {
    private CMemory<CTextureData> texBuffer = null;
    private int[] texIntBuffer = null;    
    private CMemory<IntValue> count = null;
    private int[] countIntBuffer = null;
    private TracerAPI api = null;
    
    
    
    public CTextureApplyPass(TracerAPI api, CMemory<CTextureData> texBuffer, CMemory<IntValue> count) {
        this.texBuffer = texBuffer;
        this.count = count;
        this.texIntBuffer = (int[]) texBuffer.getBufferArray();
        this.countIntBuffer = (int[]) count.getBufferArray();
        this.api = api;
    }
   
    public void process()
    {
        texBuffer.transferFromDevice();
        count.transferFromDevice();
       
        for(int index = 0; index < countIntBuffer[0]; index++)
        {
            if(hasBaseTexture(index))
            {
                Image image = api.get(getMaterialIndex(index)).param1.texture.get(); //TO CORRECT/UPDATE
                float u = getU(index);
                float v = getV(index);
                int ui = (int)(u * image.getWidth());
                int vi = (int)(v * image.getHeight());
                setArgb(image, index, ui, vi);
            }
        }
        texBuffer.transferToDevice();
    }
    
    public int getMaterialIndex(int index)
    {
        int i = getArrayIndex(index);
        return texIntBuffer[i + 10]; //index of mat >= 0
    }
    
    public boolean hasBaseTexture(int index)
    {
        int i = getArrayIndex(index);
        return texIntBuffer[i + 9] > 0;
    }
    
    public void setArgb(Image image, int index, int ui, int vi)
    {
        int i = getArrayIndex(index);
        texIntBuffer[i + 2] = image.getPixelReader().getArgb(ui, vi); //argb value
    }
    
    public float getU(int index)
    {
        int i = getArrayIndex(index);
        return Float.intBitsToFloat(texIntBuffer[i + 0]); //u
    }
    
    public float getV(int index)
    {
        int i = getArrayIndex(index);
        return Float.intBitsToFloat(texIntBuffer[i + 1]); //v
    }
    
    private int getArrayIndex(int index)
    {
        return index * 12;
    }
}
