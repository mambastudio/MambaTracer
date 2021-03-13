/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.struct;

import coordinate.struct.structbyte.Structure;

/**
 *
 * @author user
 */
public class CLightInfo extends Structure{
    public int faceId;
    public int type;
    
    public CLightInfo()
    {
        faceId = 0;
        type = 0;
    }
}
