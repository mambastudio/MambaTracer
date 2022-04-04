/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import filesystem.core.file.FileObject;
import filesystem.core.file.FileObject.ExploreType;
import static filesystem.core.file.FileObject.ExploreType.FILE;
import filesystem.explorer.FileExplorer;
import java.util.Arrays;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.StackPane;
import jfx.dialog.DialogContent;
import jfx.dialog.DialogExtend;

/**
 *
 * @author user
 */
public class FileChooser extends DialogExtend<FileObject>{
    private FileExplorer explorer = null;
    private int width, height;
    
    public FileChooser()
    {
        this(FILE, 600, 450);
    }
    
    public FileChooser(ExploreType exploreType)
    {
        this(exploreType, 600, 450);
        
    }
    
    public FileChooser(ExploreType exploreType, int width, int height)
    {
        
        explorer = new FileExplorer(exploreType);
        this.width = width;
        this.height = height;
        setup();
    }
    
    public void addExtensions(FileExplorer.ExtensionFilter... filters)
    {
        explorer.addExtensions(filters);
    }
       

    @Override
    public void setup() {
        //dialog content
        DialogContent<Boolean> settingContent = new DialogContent<>();
        settingContent.setContent(explorer, DialogContent.DialogStructure.HEADER_FOOTER);
        
        //dialog pane (main window)
        init(
                settingContent,                
                Arrays.asList(
                        new ButtonType("Ok", ButtonBar.ButtonData.OK_DONE),
                        new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE)), 
                width, height, 
                false);
        
        //set buttons and click type return
        this.setSupplier((buttonType)->{
            if(buttonType == OK)
                if(explorer.getExploreType() == FILE)
                    return explorer.getSelectedFile();
                else
                    return explorer.getSelectedFolder();
            else
                return null;
        });
    }
}
