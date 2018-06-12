/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct.array;

import cl.core.Callable;
import coordinate.struct.IntStruct;
import coordinate.struct.StructIntArray;
import wrapper.core.CBufferFactory;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 * @param <T>
 */
public class CStructIntArray<T extends IntStruct> {
    OpenCLPlatform configuration;
    StructIntArray<T> structArray;
    CIntBuffer      cbuffer;
    
    public CStructIntArray(OpenCLPlatform configuration, Class<T> clazz, int size, String name, long flag)
    {
        this.configuration = configuration;
        this.structArray = new StructIntArray(clazz, size);
        this.cbuffer = CBufferFactory.wrapInt(name, configuration.context(), configuration.queue(), structArray.getArray(), flag); 
    }
    
    public void transferFromBufferToDevice()
    {
        cbuffer.mapWriteBuffer(configuration.queue(), buf -> {});
    }
    
    public void transferFromDeviceToBuffer()
    {
        cbuffer.mapReadBuffer(configuration.queue(), buf -> {});
    }
    
    public CIntBuffer getCBuffer()
    {
        return cbuffer;
    }
    
    public StructIntArray getStructArray()
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
