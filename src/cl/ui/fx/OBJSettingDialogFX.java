/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.ui.fx;

import coordinate.parser.obj.OBJInfo;
import coordinate.parser.obj.OBJInfo.SplitOBJPolicy;
import java.util.HashMap;
import java.util.Map;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import static javafx.scene.control.ButtonType.CANCEL;
import static javafx.scene.control.ButtonType.OK;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.VBox;
import jfx.dialog.DialogAbstract;
import jfx.form.Setting;
import jfx.form.SimpleSetting;

/**
 *
 * @author user
 */
public class OBJSettingDialogFX extends DialogAbstract<Boolean> {
    
    public OBJSettingDialogFX(OBJInfo info)
    {
        //this.setContent(label);
        this.setSize(300, 340);
        this.setButtons(OK, CANCEL);
        
        this.setSupplier((buttonType)->{
            return buttonType == OK;
        });
        
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
        this.setContent(vbox);
        
        this.setSupplier((buttonType)->{
            return buttonType == OK && info.f() > 0;
        });        
    }
}