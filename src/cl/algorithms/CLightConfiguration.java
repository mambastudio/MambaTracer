/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.algorithms;

import cl.scene.CMesh;
import cl.struct.CLightInfo;
import wrapper.core.CKernel;
import wrapper.core.CMemory;
import static wrapper.core.CMemory.READ_WRITE;
import wrapper.core.OpenCLConfiguration;
import wrapper.core.memory.values.IntValue;

/**
 *
 * @author user
 */
public class CLightConfiguration {
    //Opencl configuration
    OpenCLConfiguration configuration;
    
    CMesh cmesh;
    CEnvironment cenvmap;
    
    CMemory<CLightInfo> lightInfoList = null;
    CMemory<IntValue> lightPredicate = null;
    
    PrefixSumInteger lightcount = null;
    
    CKernel identifyMeshLightsKernel = null;
    CKernel prepareLightInfoKernel = null;
    
    public CLightConfiguration(OpenCLConfiguration configuration)
    {
        this.configuration = configuration;
    }
    
    public void initLights(CMesh cmesh, CEnvironment cenvmap)
    {
        this.cmesh = cmesh;
        this.cenvmap = cenvmap;
        
        //init cl variables
        lightPredicate = configuration.createBufferI(IntValue.class, cmesh.getCount(), READ_WRITE);        
        lightcount = new PrefixSumInteger(configuration, lightPredicate);
        
        //init cl kernel
        identifyMeshLightsKernel = configuration.createKernel("identifyMeshLights", cmesh.clFaces(), cmesh.clSize(), cmesh.clMaterials(), lightPredicate);
        
        //execute kernels
        configuration.execute1DKernel(identifyMeshLightsKernel, cmesh.getCount(), 100);
        lightcount.execute();
        
        int count = lightcount.getCount();
        if(hasAreaLight())
        {
            //create light info list buffer
            lightInfoList = configuration.createBufferB(CLightInfo.class, count, READ_WRITE);
            
                        
            //init cl kernel
            prepareLightInfoKernel   = configuration.createKernel("prepareLightInfo", lightPredicate, lightcount.getPrefixSum(), cmesh.clSize(), lightInfoList);
            configuration.execute1DKernel(prepareLightInfoKernel, cmesh.getCount(), 100);
        }
    }
    
    public int getAreaLightCount()
    {
        int count = lightcount.getCount();        
        return count;
    }
    
    public boolean hasAreaLight()
    {
        if(lightcount == null)
            return false;
        else
            return getAreaLightCount() > 0;
    }
    
    public boolean hasInfiniteLight()
    {
        if(cenvmap == null)
            return false;
        else
            return cenvmap.isPresent();
    }
        
    public CMemory<CLightInfo> getCLightInfoList()
    {
        if(hasAreaLight())
            return lightInfoList;
        else
        {
            CMemory<CLightInfo> lightInfo = configuration.createBufferB(CLightInfo.class, 1, READ_WRITE);
            lightInfo.setCL(new CLightInfo());
            return lightInfo;
        }
    }
    
    public CMemory<IntValue> getCLightCount()
    {
        if(hasAreaLight())
            return lightcount.getCTotal();
        else
        {
            CMemory<IntValue> lightTotal = configuration.createValueI(IntValue.class, new IntValue(0), READ_WRITE);            
            return lightTotal;
        }
    }
    
    
}
