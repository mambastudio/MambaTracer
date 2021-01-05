/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.material.type;

import cl.abstracts.MaterialInterface.BRDFType;
import static cl.abstracts.MaterialInterface.BRDFType.EMITTER;
import jfx.form.PropertyNode;
import cl.ui.fx.material.SurfaceParameterFX;
import cl.ui.fx.material.SurfaceParameterFX;

/**
 *
 * @author user
 */
public class EmitterFX implements AbstractBrdfFX{

    @Override
    public BRDFType getType() {
        return EMITTER;
    }

    @Override
    public PropertyNode getPropertyNode() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void bindSurfaceParameter(SurfaceParameterFX param) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
