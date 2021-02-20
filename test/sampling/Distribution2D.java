/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling;

import static coordinate.utility.Utility.clamp;

/**
 *
 * @author user
 */
public class Distribution2D {
    private final Distribution1D[] pConditionalV;
    private final Distribution1D pMarginal;
    
    public Distribution2D(float[] func, int nu, int nv)
    {
        pConditionalV = new Distribution1D[nv];
        for (int v = 0; v < nv; ++v) {
            // Compute conditional sampling distribution for $\tilde{v}$
            pConditionalV[v] = new Distribution1D(func, v * nu, nu);
        }
         // Compute marginal sampling distribution $p[\tilde{v}]$
        float[] marginalFunc = new float[nv];
        for (int v = 0; v < nv; ++v) {
            marginalFunc[v] = pConditionalV[v].funcIntegral;
        }
        
        pMarginal = new Distribution1D(marginalFunc, 0, nv);
    }
    
    public void sampleContinuous(float u0, float u1, float[] uv,
            float[] pdf)
    {
        float[] pdfs = new float[2];
        int[] v = new int[1];
        uv[1] = pMarginal.sampleContinuous(u1, pdf, v);
        pdfs[1] = pdf[0];
        
        uv[0] = pConditionalV[v[0]].sampleContinuous(u0, pdf);
        pdfs[0] = pdf[0];
        pdf[0] = pdfs[0] * pdfs[1];
    }
    
    public float pdf(float u, float v) {
        
        int iu = clamp((int) (u * pConditionalV[0].count), 0,
                pConditionalV[0].count - 1);
        int iv = clamp((int) (v * pMarginal.count), 0,
                pMarginal.count - 1);
        
        if (pConditionalV[iv].funcIntegral * pMarginal.funcIntegral == 0.f) 
            return 0.f;
       
        return (pConditionalV[iv].func[iu] * pMarginal.func[iv])
                / (pConditionalV[iv].funcIntegral * pMarginal.funcIntegral);
    }
}
