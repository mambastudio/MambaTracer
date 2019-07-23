/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import cl.core.Overlay;
import static cl.core.api.MambaAPIInterface.DeviceType.RAYTRACE;
import cl.core.device.RayDeviceMesh;
import cl.core.console.CMDConsole;
import cl.main.TracerAPI;
import filesystem.core.OutputFactory;
import java.util.function.Supplier;

/**
 *
 * @author user
 */
public class SimpleTest {
    public static void main(String... args)
    {
        //OutputFactory.setOutput(new CMDConsole());
        
        //TracerAPI api = new TracerAPI();
        //api.set(RAYTRACE, new RayDeviceMesh());
        
        Joe joe = new Joe();
        Overlay overlay = joe.getObject(() -> {
            return new Overlay(2, 2);
        });
        System.out.println(overlay.toString());
    }
    
    static class Joe
    {
        public <T> T getObject(Supplier<T> supply)
        {
            return supply.get();
        }
    }
}
