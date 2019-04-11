/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.data.struct.CMaterial;
import cl.core.data.struct.MaterialC;
import coordinate.parser.attribute.MaterialT;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author user
 */
public class TestSerializable {
    public static void main(String... args) throws IOException
    {
        
        MaterialT m = new MaterialT();
        System.out.println(isSerializable(m));
    }
    
    private static<T> boolean isSerializable(T obj)        
    {
        ObjectOutputStream oos = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            oos = new ObjectOutputStream(baos);       
            oos.writeObject(obj);
            oos.close();
            return true;
        } catch (IOException ex) {
            return false;
        } finally {
            try {
                oos.close();
            } catch (IOException ex) {
                Logger.getLogger(TestSerializable.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

}
