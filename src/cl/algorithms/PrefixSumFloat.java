/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import static cl.abstracts.MambaAPIInterface.getGlobal;
import static cl.abstracts.MambaAPIInterface.getNumOfGroups;
import java.util.Arrays;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import static wrapper.core.memory.LocalMemory.LOCALFLOAT;
import wrapper.core.memory.values.FloatValue;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class PrefixSumFloat {
    
    private final OpenCLConfiguration configuration;
    
    private final CMemory<FloatValue> cdata;
    
    private final int length;
    
    private CMemory<IntValue> clength;       
    private CMemory<FloatValue> cprefixsum;      

    private CMemory<FloatValue> cgroupSum;       
    private CMemory<FloatValue> cgroupPrefixSum;
    private CMemory<IntValue>   cgroupSize;   
    
    private CMemory<FloatValue> ctotal;
    
    private CKernel localScanFloatKernel;  
    private CKernel groupScanFloatKernel;  
    private CKernel globalScanFloatKernel;
    private CKernel globalTotalFloatKernel;
    
    private final int LOCALSIZE   = 128;
    private int GLOBALSIZE  = 0;
    private int GROUPSIZE   = 0;
    
    public PrefixSumFloat(OpenCLConfiguration configuration, CMemory<FloatValue> cdata)
    {
        this.configuration = configuration;
        this.cdata = cdata;
        this.length = cdata.getSize();
        
        GLOBALSIZE  = getGlobal(length, LOCALSIZE); //should be a power of 2 based on local size
        GROUPSIZE   = getNumOfGroups(GLOBALSIZE, LOCALSIZE);
        
        initBuffers();
        initKernels();
    }
    
    private void initBuffers()
    {
        clength           = configuration.createValueI(IntValue.class, new IntValue(length), READ_WRITE);
        cprefixsum        = configuration.createBufferF(FloatValue.class, length, READ_WRITE);
        
        cgroupSum         = configuration.createBufferF(FloatValue.class, GROUPSIZE, READ_WRITE);
        cgroupPrefixSum   = configuration.createBufferF(FloatValue.class, GROUPSIZE, READ_WRITE);
        cgroupSize        = configuration.createValueI(IntValue.class, new IntValue(GROUPSIZE), READ_WRITE);
    
        ctotal            = configuration.createValueF(FloatValue.class, new FloatValue(0), READ_WRITE);
    }
    
    private void initKernels()
    {
        localScanFloatKernel      = configuration.createKernel("localScanFloat", cdata, cprefixsum, cgroupSum, clength, LOCALFLOAT);
        groupScanFloatKernel      = configuration.createKernel("groupScanFloat", cgroupSum, cgroupPrefixSum, cgroupSize);
        globalScanFloatKernel     = configuration.createKernel("globalScanFloat", cprefixsum, cgroupPrefixSum, clength);
        globalTotalFloatKernel    = configuration.createKernel("globalTotalFloat", cdata, cprefixsum, ctotal, clength);
    }
    
    public void execute()
    {
        //these three kernels would provide a prefix sum that is quite fast for a huge array
        configuration.execute1DKernel(localScanFloatKernel, GLOBALSIZE, LOCALSIZE); //start doing a local (workgroup) size scan
        configuration.execute1DKernel(groupScanFloatKernel, 1, 1); //Do group level scan. Quite fast. Don't underestimate opencl loops
        configuration.execute1DKernel(globalScanFloatKernel, GLOBALSIZE, LOCALSIZE); //add the group level scan to the local size scan
        configuration.execute1DKernel(globalTotalFloatKernel, 1, 1);
    }
    
    public CMemory<FloatValue> getCTotal()
    {
        return ctotal;
    }
    
    public void printlnResults()
    {
        cprefixsum.transferFromDevice();        
        float[] output = (float[]) cprefixsum.getBufferArray();
        
        cgroupPrefixSum.transferFromDevice();
        float[] groupsum = (float[]) cgroupPrefixSum.getBufferArray();
        
        System.out.println(Arrays.toString(output));
        System.out.println(Arrays.toString(groupsum));
        System.out.println(ctotal.getCL().v);
    }
}
