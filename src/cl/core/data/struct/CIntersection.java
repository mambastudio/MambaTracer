/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import cl.core.data.CPoint2;
import cl.core.data.CPoint3;
import coordinate.generic.raytrace.AbstractIntersection;
import coordinate.struct.ByteStruct;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 *
 * @author user
 */
public class CIntersection extends ByteStruct implements AbstractIntersection {
    
    public CPoint3 throughput;
    public CPoint3 p;
    public CPoint3 n;
    public CPoint3 d;
    public CPoint2 pixel;
    public CPoint2 uv;
    public int mat;
    public int sampled_brdf;
    public int id;
    public int hit;   
    
    public CIntersection()
    {
        this.throughput = new CPoint3();
        this.p = new CPoint3();
        this.n = new CPoint3();
        this.d = new CPoint3();
        this.pixel = new CPoint2();
        this.uv = new CPoint2();
        this.mat = 0;
        this.sampled_brdf = 0;
        this.id = 0;
        this.hit = 0;
    }
    
    public void setMat(int mat)
    {
        this.mat = mat;
        this.refreshGlobalArray();
    }

    public void setHit(int hit)
    {
        this.hit = hit;
        this.refreshGlobalArray();
    }
    
    public void setSampledBRDF(int sample)
    {
        this.sampled_brdf = sample;
        this.refreshGlobalArray();
    }
    
    public int getHit()
    {
        return hit;
    }
    
    @Override
    public void initFromGlobalArray() {
        ByteBuffer buffer = this.getLocalByteBuffer(ByteOrder.nativeOrder()); //main buffer but position set to index and limit to size of struct
        int[] offsets = this.getOffsets();
        int pos = buffer.position();
        
        buffer.position(pos + offsets[0]);
        throughput.x = buffer.getFloat(); 
        throughput.y = buffer.getFloat(); 
        throughput.z = buffer.getFloat();
        
        buffer.position(pos + offsets[1]);
        p.x = buffer.getFloat(); 
        p.y = buffer.getFloat(); 
        p.z = buffer.getFloat();
        
        buffer.position(pos + offsets[2]);
        n.x = buffer.getFloat(); 
        n.y = buffer.getFloat(); 
        n.z = buffer.getFloat();
        
        buffer.position(pos + offsets[3]);
        d.x = buffer.getFloat(); 
        d.y = buffer.getFloat(); 
        d.z = buffer.getFloat();
                
        buffer.position(pos + offsets[4]);
        pixel.x = buffer.getFloat(); 
        pixel.y = buffer.getFloat();
                
        buffer.position(pos + offsets[5]);
        uv.x = buffer.getFloat(); 
        uv.y = buffer.getFloat(); 
        
        buffer.position(pos + offsets[6]);
        mat = buffer.getInt();
        
        buffer.position(pos + offsets[7]);
        sampled_brdf = buffer.getInt();
        
        buffer.position(pos + offsets[8]);
        id = buffer.getInt();
        
        buffer.position(pos + offsets[9]);
        hit = buffer.getInt();
    }

    @Override
    public byte[] getArray() {
        ByteBuffer buffer = this.getEmptyLocalByteBuffer(ByteOrder.nativeOrder());            
        int[] offsets = this.getOffsets();
        int pos = buffer.position(); 
        
        buffer.position(pos + offsets[0]);
        buffer.putFloat(throughput.x); 
        buffer.putFloat(throughput.y); 
        buffer.putFloat(throughput.z);
        
        buffer.position(pos + offsets[1]);        
        buffer.putFloat(p.x); 
        buffer.putFloat(p.y); 
        buffer.putFloat(p.z);
        
        buffer.position(pos + offsets[2]);
        buffer.putFloat(n.x); 
        buffer.putFloat(n.y); 
        buffer.putFloat(n.z);
        
        buffer.position(pos + offsets[3]);
        buffer.putFloat(d.x); 
        buffer.putFloat(d.y);
        buffer.putFloat(d.z);
        
        buffer.position(pos + offsets[4]);
        buffer.putFloat(pixel.x); 
        buffer.putFloat(pixel.y);
        
        buffer.position(pos + offsets[5]);
        buffer.putFloat(uv.x); 
        buffer.putFloat(uv.y);
        
        buffer.position(pos + offsets[6]);
        buffer.putInt(mat);
        
        buffer.position(pos + offsets[7]);
        buffer.putInt(sampled_brdf);
        
        buffer.position(pos + offsets[8]);
        buffer.putInt(id);
        
        buffer.position(pos + offsets[9]);
        buffer.putInt(hit); 
        
        return buffer.array();
    }
}
