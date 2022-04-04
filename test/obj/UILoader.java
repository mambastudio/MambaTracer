/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package obj;

import cl.ui.fx.OBJSettingDialogFX;
import coordinate.parser.obj.OBJInfo;
import java.io.File;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class UILoader extends Application {
    
    FileChooser chooser = new FileChooser();
    
    @Override
    public void start(Stage primaryStage) {
        Button btn = new Button("Load OBJ Wavefront");
        
        StackPane root = new StackPane();
        root.getChildren().add(btn);
        
         // Set extension filter
        FileChooser.ExtensionFilter extFilter = 
                new FileChooser.ExtensionFilter("OBJ files (*.obj)", "*.obj");
        chooser.getExtensionFilters().add(extFilter);
        
        
        btn.setOnAction((ActionEvent event) -> {
           File file = chooser.showOpenDialog(primaryStage);
           if(file != null)
           {
                OBJInfo info = new OBJInfo(file.toURI());
                info.read();
                OBJSettingDialogFX dialog = new OBJSettingDialogFX(info);
                dialog.showAndWait(primaryStage);
                System.out.println(info);
                
           }
        });        
        Scene scene = new Scene(root, 600, 500);
        
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }
    
}
