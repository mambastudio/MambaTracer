/* 
 * The MIT License
 *
 * Copyright 2016 user.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package cl.core.env.sun.data;

import cl.core.data.CVector3;
import java.util.Arrays;

/**
 *
 * @author user
 */
public class ArHosekSkyModelState {
    public ArHosekSkyModelConfiguration[]  configs     = new ArHosekSkyModelConfiguration[11];
    public double []    radiances   = new double[11];
    public double       turbidity;
    public double       solar_radius;
    public double []    emission_correction_factor_sky = new double[11];
    public double []    emission_correction_factor_sun = new double[11];
    public double       albedo;
    public double       elevation;
        
    public ArHosekSkyModelState()
    {
        for(int i = 0; i<configs.length; i++)
            configs[i] = new ArHosekSkyModelConfiguration();
    }
        
    public String radiancesString()
    {
        return Arrays.toString(radiances);
    }
        
    public String configsString()
    {
    StringBuilder builder = new StringBuilder();
        for(ArHosekSkyModelConfiguration config: configs)
        {
            builder.append(config.toString()).append("\n");
        }
        return builder.toString();
    }
    
    public double solarRadiusToDegrees()
    {
        double solarRadiusDegrees = Math.toDegrees(solar_radius);
        return solarRadiusDegrees;
    }
    
    public double elevationToDegrees()
    {
        double elevationDegrees = Math.toDegrees(elevation);
        return elevationDegrees;
    }
    
    public CVector3 getSolarDirection()
    {       
        return SphericalCoordinate.elevationRadians((float)elevation);        
    }
    
    public ArHosekSkyModelState copy() 
    {
        ArHosekSkyModelState state = new ArHosekSkyModelState();
        System.arraycopy(configs, 0, state.configs, 0, configs.length);
        System.arraycopy(radiances, 0, state.radiances, 0, radiances.length);
        System.arraycopy(emission_correction_factor_sky, 0, state.emission_correction_factor_sky, 0, emission_correction_factor_sky.length);
        System.arraycopy(emission_correction_factor_sun, 0, state.emission_correction_factor_sun, 0, emission_correction_factor_sun.length);
        state.albedo = albedo;
        state.elevation = elevation;
        state.solar_radius = solar_radius;
        state.turbidity = turbidity;
        return state;
    }
}
