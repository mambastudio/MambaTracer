/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.api;

import javafx.fxml.Initializable;

/**
 *
 * @author user
 * @param <A>
 * @param <D>
 */
public interface RenderControllerInterface<A extends MambaAPIInterface, D extends RayDeviceInterface> extends Initializable {
    public void setAPI(A api);
}
