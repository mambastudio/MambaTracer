/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core;

import org.jocl.struct.CLTypes.cl_float16;
import org.jocl.struct.Struct;

/**
 *
 * @author user
 */
public class CTransform extends Struct{
    public cl_float16 m;
    public cl_float16 mInv;
    
    public CTransform()
    {
        m.set(0 , 1); m.set(1 , 0); m.set(2 , 0); m.set(3 , 0);
        m.set(4 , 0); m.set(5 , 1); m.set(6 , 0); m.set(7 , 0);
        m.set(8 , 0); m.set(9 , 0); m.set(10, 1); m.set(11, 0);
        m.set(12, 0); m.set(13, 0); m.set(14, 0); m.set(15, 1);
        
        mInv.set(0 , 1); mInv.set(1 , 0); mInv.set(2 , 0); mInv.set(3 , 0);
        mInv.set(4 , 0); mInv.set(5 , 1); mInv.set(6 , 0); mInv.set(7 , 0);
        mInv.set(8 , 0); mInv.set(9 , 0); mInv.set(10, 1); mInv.set(11, 0);
        mInv.set(12, 0); mInv.set(13, 0); mInv.set(14, 0); mInv.set(15, 1);
    }
    
    public void setTransform(float[] m, float[] mInv)
    {
        this.m.set(0 , m[0 ]); this.m.set(1 , m[1 ]); this.m.set(2 , m[2 ]); this.m.set(3 , m[3 ]);
        this.m.set(4 , m[4 ]); this.m.set(5 , m[5 ]); this.m.set(6 , m[6 ]); this.m.set(7 , m[7 ]);
        this.m.set(8 , m[8 ]); this.m.set(9 , m[9 ]); this.m.set(10, m[10]); this.m.set(11, m[11]);
        this.m.set(12, m[12]); this.m.set(13, m[13]); this.m.set(14, m[14]); this.m.set(15, m[15]);    
        
        this.mInv.set(0 , mInv[0 ]); this.mInv.set(1 , mInv[1 ]); this.mInv.set(2 , mInv[2 ]); this.mInv.set(3 , mInv[3 ]);
        this.mInv.set(4 , mInv[4 ]); this.mInv.set(5 , mInv[5 ]); this.mInv.set(6 , mInv[6 ]); this.mInv.set(7 , mInv[7 ]);
        this.mInv.set(8 , mInv[8 ]); this.mInv.set(9 , mInv[9 ]); this.mInv.set(10, mInv[10]); this.mInv.set(11, mInv[11]);
        this.mInv.set(12, mInv[12]); this.mInv.set(13, mInv[13]); this.mInv.set(14, mInv[14]); this.mInv.set(15, mInv[15]);
    }
}
