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
import static wrapper.core.memory.LocalMemory.LOCALINT;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class PrefixSumInteger {
    private final OpenCLConfiguration configuration;
    
    private final CMemory<IntValue> cdata;
    
    private final int length;
    
    private CMemory<IntValue> clength;       
    private CMemory<IntValue> cprefixsum;      

    private CMemory<IntValue> cgroupSum;       
    private CMemory<IntValue> cgroupPrefixSum;
    private CMemory<IntValue> cgroupSize;   
    
    private CMemory<IntValue> ctotal;
    
    private CKernel localScanIntegerKernel;  
    private CKernel groupScanIntegerKernel;  
    private CKernel globalScanIntegerKernel;
    private CKernel globalTotalIntegerKernel;
    
    private final int LOCALSIZE   = 128;
    private int GLOBALSIZE  = 0;
    private int GROUPSIZE   = 0;
    
    public PrefixSumInteger(OpenCLConfiguration configuration, CMemory<IntValue> cdata)
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
        cprefixsum        = configuration.createBufferI(IntValue.class, length, READ_WRITE);
        
        cgroupSum         = configuration.createBufferI(IntValue.class, GROUPSIZE, READ_WRITE);
        cgroupPrefixSum   = configuration.createBufferI(IntValue.class, GROUPSIZE, READ_WRITE);
        cgroupSize        = configuration.createValueI(IntValue.class, new IntValue(GROUPSIZE), READ_WRITE);
    
        ctotal            = configuration.createValueI(IntValue.class, new IntValue(0), READ_WRITE);
    }
    
    private void initKernels()
    {
        localScanIntegerKernel      = configuration.createKernel("localScanInteger", cdata, cprefixsum, cgroupSum, clength, LOCALINT);
        groupScanIntegerKernel      = configuration.createKernel("groupScanInteger", cgroupSum, cgroupPrefixSum, cgroupSize);
        globalScanIntegerKernel     = configuration.createKernel("globalScanInteger", cprefixsum, cgroupPrefixSum, clength);
        globalTotalIntegerKernel    = configuration.createKernel("globalTotalInteger", cdata, cprefixsum, ctotal, clength);
    }
    
    public void execute()
    {
        //these three kernels would provide a prefix sum that is quite fast for a huge array
        configuration.execute1DKernel(localScanIntegerKernel, GLOBALSIZE, LOCALSIZE); //start doing a local (workgroup) size scan
        configuration.execute1DKernel(groupScanIntegerKernel, 1, 1); //Do group level scan. Quite fast. Don't underestimate opencl loops
        configuration.execute1DKernel(globalScanIntegerKernel, GLOBALSIZE, LOCALSIZE); //add the group level scan to the local size scan
        configuration.execute1DKernel(globalTotalIntegerKernel, 1, 1);
    }
    
    public CMemory<IntValue> getCTotal()
    {
        return ctotal;
    }
    
    public int getCount()
    {
        return ctotal.getCL().v;
    }
    
    public CMemory<IntValue> getPrefixSum()
    {
        return cprefixsum;
    }
    
    public void printlnResults()
    {
        cprefixsum.transferFromDevice();        
        int[] output = (int[]) cprefixsum.getBufferArray();
        
        cgroupPrefixSum.transferFromDevice();
        int[] groupsum = (int[]) cgroupPrefixSum.getBufferArray();
        
        System.out.println(Arrays.toString(output));
        System.out.println(Arrays.toString(groupsum));
        System.out.println(ctotal.getCL().v);
    }
}
