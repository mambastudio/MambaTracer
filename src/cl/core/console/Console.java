/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.console;

import filesystem.core.OutputInterface;
import javafx.application.Platform;
import javafx.scene.control.TextArea;

/**
 *
 * @author user
 */
public class Console implements OutputInterface{
    
    TextArea timeConsole; 
    TextArea sceneConsole; 
    TextArea performanceConsole;
    
    private String sceneParsingTime = "---";
    private String bvhBuildingTime = "---";
    private String renderingTime = "---";
    
    private String primitivesUnique = "0";
    private String eye = "---";
    private String dir = "---";
    private String fov = "---";
       
    private String deviceName = "---";
    private String deviceType = "---";
    private String deviceVendor = "---";
    private String deviceSpeed = "---";
    private String raysPerSecond = "0";
    
    public Console(TextArea timeConsole, TextArea sceneConsole, TextArea performanceConsole)
    {
        this.timeConsole = timeConsole;
        this.sceneConsole = sceneConsole;
        this.performanceConsole = performanceConsole;
        display();
    }
    
    
    @Override
    public void print(String key, String string) {
        switch (key) {
            case "scene parse time":
                sceneParsingTime = string;
                break;
            case "bvh build time":
                bvhBuildingTime = string;
                break;
            case "render":
                renderingTime = string;
                break;
            case "name":
                deviceName = string;
                break;
            case "type":
                deviceType = string;
                break;
            case "speed":
                deviceSpeed = string;
                break;
            case "vendor":
                deviceVendor = string;
                break;    
            case "eye":
                eye = string;                
                break; 
            case "dir":
                dir = string;                
                break; 
            case "fov":
                fov = string;                
                break; 
                
            default:
                break;
        }  
        Platform.runLater(() -> display());
       
    }
    
    public final void display()
    {
        StringBuilder timeStringBuilder = new StringBuilder();
        timeStringBuilder.append( "Scene parsing     : ").append(sceneParsingTime).append("\n");
        timeStringBuilder.append( "BVH building      : ").append(bvhBuildingTime).append("\n");
        timeStringBuilder.append( "Rendering         : ").append(renderingTime);
        timeConsole.setText(timeStringBuilder.toString());
        
        StringBuilder sceneStringBuilder = new StringBuilder();
        sceneStringBuilder.append("Prim id : ").append(primitivesUnique).append("\n");        
        sceneStringBuilder.append("Camera   ").append("\n");      
        sceneStringBuilder.append(" eye : ").append(eye).append("\n");   
        sceneStringBuilder.append(" dir : ").append(dir).append("\n");
        sceneStringBuilder.append(" fov : ").append(fov).append("\n");
        sceneConsole.setText(sceneStringBuilder.toString());
        
        StringBuilder performanceStringBuilder = new StringBuilder();
        performanceStringBuilder.append("Device          : ").append(deviceName).append("\n");
        performanceStringBuilder.append("Type            : ").append(deviceType).append("\n");
        performanceStringBuilder.append("Speed           : ").append(deviceSpeed).append(" MHz").append("\n");
        performanceStringBuilder.append("Vendor          : ").append(deviceVendor).append("\n");
        performanceStringBuilder.append("Rays/s          : ").append(bvhBuildingTime).append("\n");    
        performanceStringBuilder.append("Refresh Display : ").append(bvhBuildingTime).append("\n");    
        performanceConsole.setText(performanceStringBuilder.toString());
    }    
}
