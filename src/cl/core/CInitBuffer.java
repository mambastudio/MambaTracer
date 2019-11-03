/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import java.util.ArrayList;
import wrapper.core.CKernel;
import wrapper.core.OpenCLPlatform;
import wrapper.core.buffer.CFloatBuffer;
import wrapper.core.buffer.CIntBuffer;

/**
 *
 * @author user
 */
public class CInitBuffer {
    private final ArrayList<KernelBufferHolder> kernels;
    private final OpenCLPlatform platform;
    
    public CInitBuffer(OpenCLPlatform platform)
    {
        this.kernels = new ArrayList<>();
        this.platform = platform;
    }
    
    public void registerFloatBuffer(CFloatBuffer... buffers)
    {
        for(CFloatBuffer buffer: buffers)
        {
            CKernel kernel = platform.createKernel("InitFloatData", buffer);
            kernels.add(new KernelBufferHolder(kernel, buffer.getBufferSize()));
        }
    }
    
    public void registerIntBuffer(CIntBuffer... buffers)
    {
        for(CIntBuffer buffer: buffers)
        {
            CKernel kernel = platform.createKernel("InitIntData", buffer);
            kernels.add(new KernelBufferHolder(kernel, buffer.getBufferSize()));
        }
    }
    
    public void initAllBuffers()
    {
        kernels.forEach((holder) -> {
            platform.queue().put1DRangeKernel(holder.kernel, holder.size, 1);
        });
    }
    
    private class KernelBufferHolder
    {
        CKernel kernel;
        int size;
        
        public KernelBufferHolder(CKernel kernel, int size)
        {
            this.kernel = kernel;
            this.size = size;
        }
    }
}
