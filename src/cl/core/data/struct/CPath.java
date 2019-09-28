package cl.core.data.struct;

import cl.core.data.CPoint3;
import coordinate.struct.ByteStruct;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author user
 */
public class CPath extends ByteStruct{
    public CPoint3 throughput;
    public boolean active;
    public CBSDF bsdf;
    
    public CPath()
    {
        throughput = new CPoint3();
        active = false;
        bsdf = new CBSDF();
    }

    @Override
    public void initFromGlobalArray() {
        ByteBuffer buffer = this.getLocalByteBuffer(ByteOrder.nativeOrder()); //main buffer but position set to index and limit to size of struct
        int[] offsets = this.getOffsets();
        int pos = buffer.position();
        
        buffer.position(pos + offsets[0]);
        throughput.x = buffer.getFloat(); throughput.y = buffer.getFloat(); throughput.z = buffer.getFloat();
        
        buffer.position(pos + offsets[1]);
        active = buffer.getInt() == 1;
        
        buffer.position(pos + offsets[2]);
        bsdf.materialID = buffer.getInt();
        
        buffer.position(pos + offsets[3]);
        bsdf.frame.mX.x = buffer.getFloat(); bsdf.frame.mX.y = buffer.getFloat(); bsdf.frame.mX.z = buffer.getFloat();
        
        buffer.position(pos + offsets[4]);
        bsdf.frame.mY.x = buffer.getFloat(); bsdf.frame.mY.y = buffer.getFloat(); bsdf.frame.mY.z = buffer.getFloat();
        
        buffer.position(pos + offsets[5]);
        bsdf.frame.mZ.x = buffer.getFloat(); bsdf.frame.mZ.y = buffer.getFloat(); bsdf.frame.mZ.z = buffer.getFloat();
        
        buffer.position(pos + offsets[6]);
        bsdf.localDirFix.x = buffer.getFloat(); bsdf.localDirFix.y = buffer.getFloat(); bsdf.localDirFix.z = buffer.getFloat();
    }

    @Override
    public byte[] getArray() {
        ByteBuffer buffer = this.getEmptyLocalByteBuffer(ByteOrder.nativeOrder());            
        int[] offsets = this.getOffsets();
        int pos = buffer.position(); 
        
        buffer.position(pos + offsets[0]);
        buffer.putFloat(throughput.x); buffer.putFloat(throughput.y); buffer.putFloat(throughput.z);
        
        buffer.position(pos + offsets[1]);
        buffer.putInt(active ? 1 : 0);
        
        buffer.position(pos + offsets[2]);
        buffer.putInt(bsdf.materialID);
        
        buffer.position(pos + offsets[3]);
        buffer.putFloat(bsdf.frame.mX.x); buffer.putFloat(bsdf.frame.mX.y); buffer.putFloat(bsdf.frame.mX.z);
        
        buffer.position(pos + offsets[4]);
        buffer.putFloat(bsdf.frame.mY.x); buffer.putFloat(bsdf.frame.mY.y); buffer.putFloat(bsdf.frame.mY.z);
        
        buffer.position(pos + offsets[5]);
        buffer.putFloat(bsdf.frame.mZ.x); buffer.putFloat(bsdf.frame.mZ.y); buffer.putFloat(bsdf.frame.mZ.z);
        
        buffer.position(pos + offsets[6]);
        buffer.putFloat(bsdf.localDirFix.x); buffer.putFloat(bsdf.localDirFix.y); buffer.putFloat(bsdf.localDirFix.z);
        
        return buffer.array();
    }
}
