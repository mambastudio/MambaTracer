/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.api;

import coordinate.parser.attribute.MaterialT;
import java.util.ArrayList;
import javafx.fxml.Initializable;

/**
 *
 * @author user
 * @param <A>
 * @param <D>
 * @param <M>
 */
public interface RenderControllerInterface<A extends MambaAPIInterface, D extends RayDeviceInterface, M extends MaterialT> extends Initializable {
    public void displaySceneMaterial(ArrayList<M> materials);
    public void setAPI(A api);      
}
