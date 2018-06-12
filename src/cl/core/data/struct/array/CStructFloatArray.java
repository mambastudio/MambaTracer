/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct.array;

import cl.core.Callable;
import coordinate.struct.FloatStruct;
import coordinate.struct.StructFloatArray;
import wrapper.core.CBufferFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;

/**
 *
 * @author user
 * @param <T>
 */
public class CStructFloatArray<T extends FloatStruct> 
{
    OpenCLPlatform configuration;
    StructFloatArray<T> structArray;
    CFloatBuffer      cbuffer;
    
    public CStructFloatArray(OpenCLPlatform configuration, Class<T> clazz, int size, String name, long flag)
    {
        this.configuration = configuration;
        this.structArray = new StructFloatArray(clazz, size);
        this.cbuffer = CBufferFactory.wrapFloat(name, configuration.context(), configuration.queue(), structArray.getArray(), flag); 
    }
    
    public void transferFromBufferToDevice()
    {
        cbuffer.mapWriteBuffer(configuration.queue(), buf -> {});
    }
    
    public void transferFromDeviceToBuffer()
    {
        cbuffer.mapReadBuffer(configuration.queue(), buf -> {});
    }
    
    public CFloatBuffer getCBuffer()
    {
        return cbuffer;
    }
    
    public StructFloatArray getStructArray()
    {
        return structArray;
    }
    
    public T get(int index)
    {
        return structArray.get(index);
    }
    
    public void set(T node, int index)
    {
        structArray.set(node, index);
    }
        
    public int getSize()
    {
        return structArray.size();
    }
    
    public void index(int index, Callable<T> callable)
    {
        T t = get(index);
        callable.call(t);
        set(t, index);
    }
}
