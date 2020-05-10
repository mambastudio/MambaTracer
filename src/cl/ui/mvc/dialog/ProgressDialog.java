/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.mvc.dialog;

import com.sun.javafx.tk.Toolkit;
import java.util.function.Supplier;
import javafx.scene.Node;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;

/**
 *
 * @author user
 * @param <R>
 */
public class ProgressDialog<R> implements MambaDialog<R> {
    StackPane parent;
    Node dialogNode;
    
    Supplier<R> supplier;
    R r = null;
    
    public ProgressDialog(StackPane parent, Node dialogNode)
    {
        this.parent = parent;
        this.dialogNode = dialogNode;
    }
    
    @Override
    public void setResultConverter(Supplier<R> supplier)
    {
        this.supplier = supplier;
    }
    
    @Override
    public void pause() {
        Toolkit.getToolkit().enterNestedEventLoop(this);
    }

    @Override
    public void resume() {
        Toolkit.getToolkit().exitNestedEventLoop(this, null);
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
    public Supplier<R> getSupplier() {
        return supplier;
    }

    @Override
    public void setReturn(R r) {
        this.r = r;              
    }

    @Override
    public R getReturn() {
        return r;
    }
}
