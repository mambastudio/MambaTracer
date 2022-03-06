/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import cl.data.CInt4;
import coordinate.struct.structint.IntStruct;

/**
 *
 * @author user
 */
public final class CTextureData extends IntStruct{
    public CInt4 diffuseTexture;       //x, y-coord, argb, has_texture (0 or 1, false or true)
    public CInt4 glossyTexture;
    public CInt4 roughnessTexture;      
    public CInt4 mirrorTexture;    
    public CInt4 parameters; //x = materialID
        
    public CTextureData()
    {
        diffuseTexture = new CInt4();
        glossyTexture = new CInt4();
        roughnessTexture = new CInt4();
        mirrorTexture = new CInt4();
        parameters = new CInt4();
    }    
}
