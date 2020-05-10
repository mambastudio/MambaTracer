/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.dialog;

import com.sun.javafx.tk.Toolkit;
import java.util.Optional;
import java.util.function.Supplier;
import javafx.application.Platform;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.layout.Pane;
import thread.model.LambdaThread;

/**
 *
 * @author user
 * @param <R>
 */
public interface MambaDialog<R>
{
    public void setResultConverter(Supplier<R> supplier);
    
    default void pause() {
        Toolkit.getToolkit().enterNestedEventLoop(this);
    }

    default void resume() {
        Toolkit.getToolkit().exitNestedEventLoop(this, null);
    }
    
    public Pane getParentNode();
    public Node getDialogNode();
    public Supplier<R> getSupplier();
    public void setReturn(R r);
    public R getReturn();
    
    default Optional<R> showAndWait()
    {
        //disable parent children and add dialog
        getParentNode().getChildren().forEach((node) -> {
            node.setDisable(true);
        });        
        Group group = new Group(getDialogNode());
        getParentNode().getChildren().add(group);
        group.toFront();
        
        
        R r = getSupplier().get();
        setReturn(r);
        pause();
        
        //remove dialog and enable parent children
        getParentNode().getChildren().remove(getParentNode().getChildren().size()-1);
        getParentNode().getChildren().forEach((node) -> {
            node.setDisable(false);
        });   
               
        return Optional.ofNullable(getReturn());
    }
    
    default Optional<R> showAndWaitThread()
    {        
        //disable parent children and add dialog
        getParentNode().getChildren().forEach((node) -> {
            node.setDisable(true);
        });        
        getParentNode().getChildren().add(new Group(getDialogNode())); 
            
        //thread execution
        LambdaThread.executeThread(()->{
            R r = getSupplier().get();
            setReturn(r);
            Platform.runLater(()-> resume());
        });
        pause();
        
        //remove dialog and enable parent children
        getParentNode().getChildren().remove(getParentNode().getChildren().size()-1);
        getParentNode().getChildren().forEach((node) -> {
            node.setDisable(false);
        });   
               
        return Optional.ofNullable(getReturn());
    }
}
