/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import java.util.concurrent.FutureTask;
import javafx.application.Platform;
import javafx.scene.Node;
import thread.model.LambdaThread;

/**
 *
 * @author user
 */
public class SingleTaskFX {
    private final LambdaThread task;
    
    public SingleTaskFX()
    {
        task = new LambdaThread();
    }
    
    public void execute(Node node, Runnable runnable)
    {
        if(isTaskPresent()) return;
                
        //execute task
        executeTask(node, runnable);
    }
    
    public boolean isTaskPresent()
    {
        return !isTerminated();
    }
    
    private void executeTask(Node node, Runnable runnable)
    {
        task.startExecution(()->{
            Platform.runLater(()->node.setOpacity(1));
            runnable.run();            
            Platform.runLater(()->node.setOpacity(0));
            task.stopExecution();
        });
       
    }
    
    private boolean isTerminated()
    {
        return task.isTerminated();
    }
}
