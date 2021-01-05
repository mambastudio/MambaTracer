/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx.main;

import bitmap.display.BlendDisplay;
import bitmap.display.ImageDisplay;
import static bitmap.display.gallery.GalleryCanvas.ImageType.HDR;
import bitmap.display.gallery.HDRImageLoaderFactory;
import static cl.abstracts.MambaAPIInterface.DeviceType.RAYTRACE;
import static cl.abstracts.MambaAPIInterface.DeviceType.RENDER;
import static cl.abstracts.MambaAPIInterface.ImageType.OVERLAY_IMAGE;
import static cl.abstracts.MambaAPIInterface.ImageType.RAYTRACE_IMAGE;
import cl.fx.GalleryDialogFX;
import cl.fx.UtilityHandler;
import filesystem.core.UI;
import java.io.IOException;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 *
 * @author user
 */
public class MambaTracer extends Application {
    
    static 
    {
        HDRImageLoaderFactory.install();
        UtilityHandler.register("environment", new GalleryDialogFX("environment", HDR));
    }

    public MambaTracer() {
        
    }
    
    @Override
    public void start(Stage primaryStage) throws IOException {        
        FXMLLoader loader = new FXMLLoader(getClass().getResource("UserInterfaceFXML.fxml"));
        Parent root = loader.load();
        
        /*START EVERYTHING HERE*/
        //create api and set display
        TracerAPI api = new TracerAPI();       
        api.setBlendDisplay(RAYTRACE, new BlendDisplay(RAYTRACE_IMAGE.name(), OVERLAY_IMAGE.name()));
        api.setBlendDisplay(RENDER, new ImageDisplay());
        
        //set controller (which sets display inside)
        UserInterfaceFXMLController controller = (UserInterfaceFXMLController)loader.getController();  
        api.set("controller", controller);
        
        UI.setConsole(controller);
        
        //init mesh, device, images
        api.init();
        
        //complete launch of ui
        Scene scene = new Scene(root);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Mamba Tracer");
        primaryStage.setMinWidth(900);
        primaryStage.setMinHeight(650);
        primaryStage.show();
        primaryStage.setOnCloseRequest(e -> {
            Platform.runLater(()->System.exit(0));
        });
       
        UtilityHandler.setScene(scene);
    }
    
    

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {        
        launch(args);
    }
    
}
