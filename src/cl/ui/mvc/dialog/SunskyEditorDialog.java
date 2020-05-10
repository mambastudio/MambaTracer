/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.dialog;

import bitmap.display.DynamicDisplay;
import cl.core.env.sun.model.SunskyRender;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;
import java.util.function.Supplier;
import javafx.event.ActionEvent;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.effect.DropShadow;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;

/**
 * FXML Controller class
 *
 * @author user
 */
public class SunskyEditorDialog extends BorderPane implements MambaDialog{
    
    StackPane parent;
    Node dialogNode;
    
    Supplier supplier;
    Object r = null;
    
    private final SunskyRender render = new SunskyRender();
    private final DynamicDisplay display = new DynamicDisplay();
    
    public SunskyEditorDialog(StackPane parent)
    {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource(
            "SunskyEditorDialog.fxml"));
        fxmlLoader.setRoot(this);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        } 
        
        this.parent = parent;
        this.dialogNode = this;
        this.setBackground(new Background(new BackgroundFill(Color.web("#f4f4f4"), CornerRadii.EMPTY, Insets.EMPTY)));
        this.setEffect(new DropShadow(10, Color.GREY));
        
        this.setCenter(display);
        render.startExecution(display);
        
        this.display.viewportW.addListener((observable, old_value, new_value) -> {            
            render.resume();
        });        
        this.display.viewportH.addListener((observable, old_value, new_value) -> {
            render.resume();
        });     
    }
    
    @Override
    public void setResultConverter(Supplier supplier) {
        this.supplier = supplier;
    }

    @Override
    public Pane getParentNode() {
        return parent;
    }

    @Override
    public Node getDialogNode() {
        return dialogNode;
    }

    @Override
    public Supplier getSupplier() {
        return supplier;
    }

    @Override
    public void setReturn(Object r) {        
        this.r = r;
    }

    @Override
    public Object getReturn() {
        return r;
    }
    
    public void exitDialog(ActionEvent e)
    {
        this.resume();
        this.render.shutdownThread();
    }
}
