/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import coordinate.parser.attribute.MaterialT;

/**
 *
 * @author user
 */
public interface CMaterialInterface {
    public void setMaterial(MaterialT mat);
    public MaterialT getMaterial();
}
