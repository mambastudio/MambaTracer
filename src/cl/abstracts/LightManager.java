/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.abstracts;

/**
 *
 * @author user
 */
public interface LightManager {
    public int getLightCount();
    public boolean hasInfiniteLight();
    public boolean hasAreaLight();
    public boolean initLight();
    
}
