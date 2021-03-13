/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling;

import static java.lang.Math.max;
import java.util.Arrays;

/**
 *
 * @author user
 */
public class Distribution1D {
    public int count;
    float funcIntegral;
    float[] func;
    float[] cdf;
    
    public Distribution1D(float[] f, int offset, int n) {
        count = n;
        func = new float[n];
        System.arraycopy(f, offset, func, 0, n);
        cdf = new float[n + 1];
        // Compute integral of step function at $x_i$
        cdf[0] = 0.f;
        for (int i = 1; i < count + 1; ++i) {
            cdf[i] = cdf[i - 1] + func[i - 1] / n;
        }

        // Transform step function integral into CDF
        funcIntegral = cdf[count];
        if (funcIntegral == 0.f) { //if func values are all 0
            for (int i = 1; i < n + 1; ++i) {
                cdf[i] = (float) (i) / (float) (n);
            }
        } else {
            for (int i = 1; i < n + 1; ++i) {
                cdf[i] /= funcIntegral;
            }
        }
    }
    
    public float sampleContinuous(float u, float[] pdf) {
        return sampleContinuous(u, pdf, null);
    }
    
    public float sampleContinuous(float u, float[] pdf, int[] off) {
        // Find surrounding CDF segments and _offset_
        int ptr = upper_bound(cdf, 0, count, u);
        int offset = max(0, ptr - 1);
        if (off != null) {
            off[0] = offset;
        }
        assert (offset < count);
        assert (u >= cdf[offset] && u < cdf[offset + 1]);

        // Compute offset along CDF segment
        float du = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
        assert (!Float.isNaN(du));

        // Compute PDF for sampled offset
        if (pdf != null) {
            pdf[0] = func[offset] / funcIntegral;
        }

        // Return $x\in{}[0,1)$ corresponding to sample
        return (offset + du) / count;
    }
    
    public int sampleDiscrete(float u, float[] pdf) { //pdf is for storing a single pdf value
        // Find surrounding CDF segments and _offset_
        int ptr = upper_bound(cdf, 0, count, u);
        int offset = max(0, ptr - 1);
        assert (offset < count);
        assert (u >= cdf[offset] && u < cdf[offset + 1]);
        if (pdf != null) { //store the pdf
            pdf[0] = func[offset] / (funcIntegral * count);
        }
        return offset;
    }
    
    public float discretePDF(int index)
    {
        return func[index]/(funcIntegral * count);
    }
    
    private static int upper_bound(float[] a, int first, int last, float value) {
        int i;
        for (i = first; i < last; i++) {
            if (a[i] > value) {
                break;
            }
        }
        return i;
    }
    
    public void printlnCDF()
    {
        System.out.println(Arrays.toString(cdf));
    }
}
