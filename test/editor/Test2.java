/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package editor;

import cl.fx.UtilityHandler;
import cl.ui.fx.material.MaterialFX2;
import cl.ui.fx.material.MaterialFX2EditorDialog;
import java.util.Optional;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class Test2 extends Application{
    
    public static void main(String... args)
    {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        
        //MaterialFX2Editor root = new MaterialFX2Editor(new MaterialFX2());
        
        
        //Optional<MaterialFX> option = DialogUtility.showAndWait(UtilityHandler.getScene(), new MaterialEditor(defMat, UtilityHandler.getGallery("texture")));
        
       // root.setVbarPolicy(ScrollPane.ScrollBarPolicy.ALWAYS);
        Button button = new Button("launch material editor");
        
        
        StackPane root = new StackPane();
        root.getChildren().add(button);
        Scene scene = new Scene(root, 800, 650);
        UtilityHandler.setScene(scene);
        
        button.setOnAction(e->{
            MaterialFX2EditorDialog materialDialog = new MaterialFX2EditorDialog(new MaterialFX2(), UtilityHandler.getGallery("texture"));
            Optional<MaterialFX2> option = materialDialog.showAndWait(UtilityHandler.getScene());
        });
        
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}
