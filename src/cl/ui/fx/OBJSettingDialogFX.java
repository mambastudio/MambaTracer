/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import coordinate.parser.obj.OBJInfo;
import coordinate.parser.obj.OBJInfo.SplitOBJPolicy;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.VBox;
import jfx.dialog.DialogContent;
import jfx.dialog.DialogExtend;
import jfx.form.Setting;
import jfx.form.SimpleSetting;

/**
 *
 * @author user
 */
public class OBJSettingDialogFX extends DialogExtend<Boolean> {
    
    double width = 300;
    double height = 340;
    
    OBJInfo info;    
    
    public OBJSettingDialogFX(OBJInfo info)
    {
        this.info = info;
        this.setup();
    }

    @Override
    public void setup() {
        //dialog content
        DialogContent<Boolean> settingContent = new DialogContent<>();
        
        //file statistics
        HashMap<String, String> stateObj = info.getInfoString();
        Setting[] fileStatsticSettings = new Setting[stateObj.size()];
        int i = 0;
        for (Map.Entry<String, String> entry : stateObj.entrySet()) 
        {
            fileStatsticSettings[i] = Setting.of(entry.getKey(), entry.getValue());
            i++;
        }
        
        //split policy
        ObservableList<SplitOBJPolicy> availableSplitPolicy = FXCollections.observableArrayList(info.getAvailableSplitPolicy());
        ObjectProperty<SplitOBJPolicy> currentSplitPolicy = new SimpleObjectProperty(info.splitPolicy());
        currentSplitPolicy.addListener((o, ov, nv)->{
            info.setSplitPolicy(nv);
        });
        Setting splitPolicySettings = Setting.of("Type", 
                        availableSplitPolicy, 
                        currentSplitPolicy);
        
        VBox vbox = new VBox();
        vbox.setSpacing(5);
        
        TitledPane fileStatistics = new TitledPane("File Statistics", SimpleSetting.createForm(0, fileStatsticSettings));
        fileStatistics.setCollapsible(false);
        TitledPane splitPolicy = new TitledPane("Split Policy", SimpleSetting.createForm(0, splitPolicySettings));
        splitPolicy.setCollapsible(false);
        
        vbox.getChildren().addAll(fileStatistics, splitPolicy);
        settingContent.setContent(vbox, DialogContent.DialogStructure.HEADER_FOOTER);
        
        //dialog pane (main window)
        init(
                settingContent,                
                Arrays.asList(
                        new ButtonType("Ok", ButtonBar.ButtonData.OK_DONE),
                        new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE)), 
                width, height, 
                false);
        
        this.setSupplier((buttonType)->{
            return buttonType == OK && info.f() > 0;
        });      
    }
}