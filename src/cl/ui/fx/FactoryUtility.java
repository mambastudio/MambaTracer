/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

/**
 *
 * @author user
 */
public class FactoryUtility {
    //https://stackoverflow.com/questions/4051202/get-the-outer-class-object-from-an-inner-class-object
    public static boolean isEnclosingClassEqual(Object object, String name)
    {
        return object.getClass().getEnclosingClass().getSimpleName().equals(name);
    }
    
}
