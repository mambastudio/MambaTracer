/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.api;

import bitmap.image.BitmapARGB;

/**
 *
 * @author user
 */
public interface MambaAPIInterface 
{
    public void setRayDevice(RayDeviceInterface device);
    public void setRenderBuffer(String name, BitmapARGB bitmap);
    public void copyToArray(String name, int[] array);
    public void setImageSize(int width, int height);
    public void rayTraceFast();
}
