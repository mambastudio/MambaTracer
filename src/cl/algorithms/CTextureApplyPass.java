/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.abstracts.MambaAPIInterface;
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
    private MambaAPIInterface api = null;
    
    private final TextureInfoCache textureInfoCache;
    
    
    public CTextureApplyPass(MambaAPIInterface api, CMemory<CTextureData> texBuffer, CMemory<IntValue> count) {
        this.texBuffer = texBuffer;
        this.count = count;
        this.texIntBuffer = (int[]) texBuffer.getBufferArray();
        this.countIntBuffer = (int[]) count.getBufferArray();
        this.api = api;
        this.textureInfoCache = new TextureInfoCache(texIntBuffer);
    }
   
    public void process()
    {
        texBuffer.transferFromDevice();
        count.transferFromDevice();
               
        for(int index = 0; index < countIntBuffer[0]; index++)
        {            
            textureInfoCache.setIndex(index);
            if(textureInfoCache.hasDiffuseTexture())
            {
                Image image = api.getMaterial(textureInfoCache.getMaterialIndex()).getDiffuseTexture(); //TO CORRECT/UPDATE
                float x = textureInfoCache.getDiffuseTextureU();
                float y = textureInfoCache.getDiffuseTextureV();
                             
                if(image != null)
                {
                    float dx = (float) x * ((float)image.getWidth() - 1);
                    float dy = (float) y * ((float)image.getHeight() - 1);

                    textureInfoCache.setDiffuseArgb(image, (int) dx, (int)dy);
                }
            }
            
            if(textureInfoCache.hasGlossyTexture())
            {   
                Image image = api.getMaterial(textureInfoCache.getMaterialIndex()).getGlossyTexture(); //TO CORRECT/UPDATE
                float x = textureInfoCache.getGlossyTextureU();
                float y = textureInfoCache.getGlossyTextureV();
                
                if(image != null)
                {
                    float dx = (float) x * ((float)image.getWidth() - 1);
                    float dy = (float) y * ((float)image.getHeight() - 1);

                    textureInfoCache.setGlossyArgb(image, (int) dx, (int)dy);
                }
            }
            
            if(textureInfoCache.hasRoughnessTexture())
            {   
                Image image = api.getMaterial(textureInfoCache.getMaterialIndex()).getRoughnessTexture(); //TO CORRECT/UPDATE
                float x = textureInfoCache.getRoughnessTextureU();
                float y = textureInfoCache.getRoughnessTextureV();
                
                if(image != null)
                {
                    float dx = (float) x * ((float)image.getWidth() - 1);
                    float dy = (float) y * ((float)image.getHeight() - 1);

                    textureInfoCache.setRoughnessArgb(image, (int) dx, (int)dy);
                }
            }
           
        }
        texBuffer.transferToDevice();
    }
}
