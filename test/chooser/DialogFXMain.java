/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package chooser;

import cl.ui.fx.FileChooser;
import filesystem.core.file.FileObject;
import static filesystem.core.file.FileObject.ExploreType.FOLDER;
import filesystem.explorer.FileExplorer.ExtensionFilter;
import java.util.Optional;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class DialogFXMain extends Application {
    FileChooser chooser = new FileChooser(FOLDER); 
    
    @Override
    public void start(Stage primaryStage) {
        Button btn = new Button();
        
        StackPane root = new StackPane();
        root.getChildren().add(btn);
        
        double r=20;
        Circle circ = new Circle(5); 
        circ.setFill(Color.CADETBLUE);
        btn.setShape(new Circle(r));
        btn.setGraphic(circ);
        btn.setMinSize(r, r);
        btn.setMaxSize(r, r);
        
        
        btn.setOnAction((ActionEvent event) -> {
            /*
            SimpleDialog dialog = new SimpleDialog();
            Optional text =  DialogUtility.showAndWait(root, dialog);
            if(text.isPresent())
                System.out.println(text.get());
            else
                System.out.println("empty");
            */
            
            Optional<FileObject> fileOptional = chooser.showAndWait(root);
            if(fileOptional.isPresent())
            {
                System.out.println(fileOptional.get().getName());
            }
            
            /*
            ProcessDialog dialog = new ProcessDialog();
            dialog.setMessage("Loading");          
            DialogUtility.showAndWaitThread(root, dialog, (type)->{
                try {
                    Thread.sleep(2000);
                    Platform.runLater(()->{dialog.setMessage("kubafu");});
                    Thread.sleep(2000);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DialogFXMain.class.getName()).log(Level.SEVERE, null, ex);
                }
                return true;
            });
            //System.out.println(DialogUtility.showAndWait(root, new TextInputDialog("Enter file name")));
            */
        });        
        Scene scene = new Scene(root, 700, 600);
        
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
        
        chooser.addExtensions(new ExtensionFilter("HDR", ".hdr"));
        
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }
    
}
