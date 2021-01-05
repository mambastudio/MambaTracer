/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material.type;

import cl.abstracts.MaterialInterface.BRDFType;
import jfx.form.PropertyNode;
import cl.ui.fx.material.SurfaceParameterFX;
import cl.ui.fx.material.SurfaceParameterFX;

/**
 *
 * @author user
 */
public interface AbstractBrdfFX {    
    public BRDFType getType();          
    public PropertyNode getPropertyNode(); 
    public void bindSurfaceParameter(SurfaceParameterFX param);
}
