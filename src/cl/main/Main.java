/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.main;

import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class Main extends Application {
    public static void main(String... args)
    {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("RenderWindow.fxml"));
        
        Scene scene = new Scene(root);        
        primaryStage.setScene(scene);
        primaryStage.setTitle("Mamba Tracer");
        primaryStage.setMinWidth(1250);
        primaryStage.setMinHeight(880);
        primaryStage.show();
        primaryStage.setOnCloseRequest(e -> {
            System.exit(0);
        });
    }
}
