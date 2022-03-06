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
import javafx.scene.control.ButtonType;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.StackPane;
import jfx.dialog.DialogAbstract;

/**
 *
 * @author user
 */
public class FileChooser extends DialogAbstract<FileObject>{
    private FileExplorer explorer = null;
    
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
        
        StackPane display = new StackPane();
        display.getChildren().add(explorer);
        
        
        this.setContent(display);
        
        this.setSize(width, height);
        
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
        this.setButtons(OK, CANCEL);
        
        //set file click in table as a return
        explorer.setTableFileClick(e->{
            if(e.getClickCount() == 2)
            {
                setButtonType(ButtonType.OK);
                resume();
            }
        });
        
        
        this.setOnKeyPressed(e->{
            if(e.getCode() == KeyCode.ENTER)
            {       
                resume();
            }
            else if(e.getCode() == KeyCode.ESCAPE)
            {                
                resume();
            }
        });
    }
    
    public void addExtensions(FileExplorer.ExtensionFilter... filters)
    {
        explorer.addExtensions(filters);
    }
       
    @Override
    public void initDefault()
    {
        explorer.setFileNull();
    }
}
