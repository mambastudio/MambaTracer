package cl.abstracts;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import coordinate.parser.attribute.MaterialT;
import java.util.ArrayList;
import javafx.fxml.Initializable;

/**
 *
 * @author user
 * @param <A>
 * @param <M>
 */
public interface RenderControllerInterface<A extends MambaAPIInterface, M extends MaterialInterface> extends Initializable {
    public enum PrintType {
        PROGRESS,
        ERROR
    }
    
    public void displaySceneMaterial(ArrayList<MaterialT> materials);
    public void setAPI(A api);        
}
