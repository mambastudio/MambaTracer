/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package editor;

import cl.ui.fx.material.SurfaceParameterFX2;
import cl.ui.fx.material.SurfaceParameterFX2Editor;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class Test extends Application{
    
    public static void main(String... args)
    {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        SurfaceParameterFX2Editor root = new SurfaceParameterFX2Editor(new SurfaceParameterFX2());
        Scene scene = new Scene(root, 600, 500);
        
        root.setFitToWidth(true);
       // root.setVbarPolicy(ScrollPane.ScrollBarPolicy.ALWAYS);
        
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
}
