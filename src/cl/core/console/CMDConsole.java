/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.console;

import filesystem.core.OutputInterface;

/**
 *
 * @author user
 */
public class CMDConsole implements OutputInterface{

    @Override
    public void print(String key, String string) {
        System.out.println(string);
    }
    
}
