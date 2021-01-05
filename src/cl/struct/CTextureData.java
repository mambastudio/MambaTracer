/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CInt4;
import coordinate.struct.IntStruct;

/**
 *
 * @author user
 */
public class CTextureData extends IntStruct{
    public CInt4 baseTexture;
    public CInt4 opacityTexture;
    public int  hasOpacity, hasBaseTex, materialID, paramLevel;
    
    public CTextureData()
    {
        this.baseTexture = new CInt4();
        this.opacityTexture = new CInt4();
        this.hasOpacity = 0;
        this.hasBaseTex = 0;
        this.materialID = 0;
        this.paramLevel = 0;
    }
}
