/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package raytrace;

import cl.abstracts.RenderControllerInterface;
import cl.ui.fx.material.MaterialFX2;
import coordinate.parser.attribute.MaterialT;
import java.net.URL;
import java.util.ArrayList;
import java.util.ResourceBundle;
import javafx.fxml.Initializable;

/**
 * FXML Controller class
 *
 * @author user
 */
public class RaytraceUIController implements Initializable, RenderControllerInterface<RaytraceAPI, MaterialFX2>{

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }    

    @Override
    public void displaySceneMaterial(ArrayList<MaterialT> materials) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setAPI(RaytraceAPI api) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
