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
package cl.core.env.sun.model;

import cl.core.env.sun.data.SphericalCoordinate;
import bitmap.CIE1931;
import bitmap.Color;
import bitmap.RGBSpace;
import bitmap.XYZ;
import cl.core.data.CVector3;
import cl.core.env.sun.data.*;
import static cl.core.env.sun.data.ArHosekSkyModelData_CIEXYZ.*;
import static cl.core.env.sun.data.ArHosekSkyModelData_RGB.*;
import static cl.core.env.sun.data.ArHosekSkyModelData_Spectral.*;
import static java.lang.Math.cos;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import java.util.Arrays;

/**
 *
 * @author user
 */
public class HosekWilkie {
    final static int NIL                            = 0;
    final static double MATH_PI                     = 3.141592653589793;
    final static double MATH_DEG_TO_RAD             = ( MATH_PI / 180.0 );
    final static double MATH_RAD_TO_DEG             = ( 180.0 / MATH_PI );
    final static double DEGREES                     = MATH_DEG_TO_RAD;
    private static double TERRESTRIAL_SOLAR_RADIUS  = ( ( 12.9 * DEGREES ) / 2.0 );
    
    private  HosekWilkie()
    {
        
    }
    
    public static void ArHosekSkyModel_CookConfiguration(
        double[]                         dataset,
        ArHosekSkyModelConfiguration  config, 
        double                         turbidity, 
        double                         albedo, 
        double                    solar_elevation
        )
    {
        double[]  elev_matrix;

        int     int_turbidity = (int)turbidity;
        double  turbidity_rem = turbidity - (double)int_turbidity;

        solar_elevation = pow(solar_elevation / (MATH_PI / 2.0), (1.0 / 3.0));
        
        // alb 0 low turb
        
        elev_matrix = Arrays.copyOfRange(dataset, 9 * 6 * (int_turbidity-1), dataset.length);
        
        for(int i = 0; i < 9; ++i )
        {
            //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
            config.values[i] = 
                    (1.0-albedo) * (1.0 - turbidity_rem) 
                    * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  + 
                    5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
                    10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
                    10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
                    5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
                    pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
        }
        
        // alb 1 low turb
        elev_matrix = Arrays.copyOfRange(dataset, 9*6*10 + 9*6*(int_turbidity-1), dataset.length);
        for(int i = 0; i < 9; ++i)
        {
            //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
            config.values[i] += 
                (albedo) * (1.0 - turbidity_rem)
                * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  + 
                5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
                10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
                10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
                5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
                pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
        }
        
        if(int_turbidity == 10)
            return;

        // alb 0 high turb
        elev_matrix = Arrays.copyOfRange(dataset, (9*6*(int_turbidity)), dataset.length);
        
        for(int i = 0; i < 9; ++i)
        {
            //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
            config.values[i] += 
                (1.0-albedo) * (turbidity_rem)
                * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  + 
                5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
                10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
                10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
                5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
                pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
        }

        // alb 1 high turb
        elev_matrix = Arrays.copyOfRange(dataset, (9*6*10 + 9*6*(int_turbidity)), dataset.length);
        for(int i = 0; i < 9; ++i)
        {
            //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
            config.values[i] += 
                (albedo) * (turbidity_rem)
                * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  + 
                5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
                10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
                10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
                5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
                pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
        }
    }
    
    static double ArHosekSkyModel_CookRadianceConfiguration(
        double[]                          dataset, 
        double                            turbidity, 
        double                            albedo, 
        double                            solar_elevation
        )
    {
        double[] elev_matrix;

        int int_turbidity = (int)turbidity;
        double turbidity_rem = turbidity - (double)int_turbidity;
        double res;
        solar_elevation = pow(solar_elevation / (MATH_PI / 2.0), (1.0 / 3.0));
        
        // alb 0 low turb
        elev_matrix = Arrays.copyOfRange(dataset, 6*(int_turbidity-1), dataset.length);
        
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        res = (1.0-albedo) * (1.0 - turbidity_rem) *
            ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
            5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
            10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
            10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
            5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
            pow(solar_elevation, 5.0) * elev_matrix[5]);

        // alb 1 low turb
        elev_matrix = Arrays.copyOfRange(dataset, 6*10 + 6*(int_turbidity-1), dataset.length);
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        res += (albedo) * (1.0 - turbidity_rem) *
            ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
            5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
            10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
            10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
            5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
            pow(solar_elevation, 5.0) * elev_matrix[5]);
        if(int_turbidity == 10)
            return res;

        // alb 0 high turb
        elev_matrix = Arrays.copyOfRange(dataset, 6*(int_turbidity), dataset.length);
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        res += (1.0-albedo) * (turbidity_rem) *
            ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
            5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
            10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
            10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
            5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
            pow(solar_elevation, 5.0) * elev_matrix[5]);

        // alb 1 high turb
        elev_matrix = Arrays.copyOfRange(dataset, 6*10 + 6*(int_turbidity), dataset.length);
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        res += (albedo) * (turbidity_rem) *
            ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
            5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
            10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
            10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
            5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
            pow(solar_elevation, 5.0) * elev_matrix[5]);
        return res;
    }
    
     
    static double ArHosekSkyModel_GetRadianceInternal(
        ArHosekSkyModelConfiguration  configuration, 
        double                        theta, 
        double                        gamma
        )
    {
        double expM = exp(configuration.values[4] * gamma);
        double rayM = cos(gamma)*cos(gamma);
        double mieM = (1.0 + cos(gamma)*cos(gamma)) / pow((1.0 + configuration.values[8]*configuration.values[8] - 2.0*configuration.values[8]*cos(gamma)), 1.5);
        double zenith = sqrt(cos(theta));

        return (1.0 + configuration.values[0] * exp(configuration.values[1] / (cos(theta) + 0.01))) *
            (configuration.values[2] + configuration.values[3] * expM + configuration.values[5] * rayM + configuration.values[6] * mieM + configuration.values[7] * zenith);
    }
    
      /* ----------------------------------------------------------------------------

        arhosekskymodelstate_alloc_init() function
        ------------------------------------------

        Initialises an ArHosekSkyModelState struct for a terrestrial setting 
        
        Spectral version
    ---------------------------------------------------------------------------- */
    public static ArHosekSkyModelState   arhosekskymodelstate_alloc_init(
        double  solar_elevation,
        double  atmospheric_turbidity,
        double  ground_albedo
        )
    {
        
        ArHosekSkyModelState state = new ArHosekSkyModelState();
        
        state.solar_radius = TERRESTRIAL_SOLAR_RADIUS;//( 0.9d * DEGREES ) / 2.0; //repetition in code with TERRESTIAL_SOLAR_RADIUS
        state.turbidity    = atmospheric_turbidity;
        state.albedo       = ground_albedo;
        state.elevation    = solar_elevation;
        
        for(int wl = 0; wl < 11; ++wl )
        {
            ArHosekSkyModel_CookConfiguration(
                datasets[wl], 
                state.configs[wl], 
                atmospheric_turbidity, 
                ground_albedo, 
                solar_elevation
            );

            state.radiances[wl] = 
            ArHosekSkyModel_CookRadianceConfiguration(
                datasetsRad[wl],
                atmospheric_turbidity,
                ground_albedo,
                solar_elevation
                );

            state.emission_correction_factor_sun[wl] = 1.0;
            state.emission_correction_factor_sky[wl] = 1.0;
        }
        return state;        
    }
    
    /* ----------------------------------------------------------------------------

    arhosekskymodelstate_alienworld_alloc_init() function
    -----------------------------------------------------

    Initialises an ArHosekSkyModelState struct for an "alien world" setting
    with a sun of a surface temperature given in 'kelvin'. The parameter
    'solar_intensity' controls the overall brightness of the sky, relative
    to the solar irradiance on Earth. A value of 1.0 yields a sky dome that
    is, on average over the wavelenghts covered in the model (!), as bright
    as the terrestrial sky in radiometric terms. 
    
    Which means that the solar radius has to be adjusted, since the 
    emissivity of a solar surface with a given temperature is more or less 
    fixed. So hotter suns have to be smaller to be equally bright as the 
    terrestrial sun, while cooler suns have to be larger. Note that there are
    limits to the validity of the luminance patterns of the underlying model:
    see the discussion above for more on this. In particular, an alien sun with
    a surface temperature of only 2000 Kelvin has to be very large if it is
    to be as bright as the terrestrial sun - so large that the luminance 
    patterns are no longer a really good fit in that case.
    
    If you need information about the solar radius that the model computes
    for a given temperature (say, for light source sampling purposes), you 
    have to query the 'solar_radius' variable of the sky model state returned 
    *after* running this function.

    ---------------------------------------------------------------------------- */

     public static ArHosekSkyModelState  arhosekskymodelstate_alienworld_alloc_init(
        float  solar_elevation,
        float  solar_intensity,
        float  solar_surface_temperature_kelvin,
        float  atmospheric_turbidity,
        float  ground_albedo
        )
    {
        return null;
    }
    
    public static void arhosekskymodelstate_free(ArHosekSkyModelState  state)
    {
        
    }
    
     //Spectral version
    
    public static double arhosekskymodel_radiance(
        ArHosekSkyModelState    state,
        double                  theta, 
        double                  gamma, 
        double                  wavelength
        )
    {
        int low_wl = (int) ((wavelength - 320.0 ) / 40.0);
        

        if ( low_wl < 0 || low_wl >= 11 )
            return 0.0f;

        double interp = ((wavelength - 320.0 ) / 40.0) % 1.0;
        
        double val_low = 
            ArHosekSkyModel_GetRadianceInternal(
                state.configs[low_wl],
                theta,
                gamma
             )
            *   state.radiances[low_wl]
            *   state.emission_correction_factor_sky[low_wl];

        if ( interp < 1e-6 )
            return val_low;

        double result = ( 1.0 - interp ) * val_low;

        if ( low_wl+1 < 11 )
        {
            result +=
                interp
                * ArHosekSkyModel_GetRadianceInternal(
                    state.configs[low_wl+1],
                    theta,
                    gamma
                  )
            * state.radiances[low_wl+1]
            * state.emission_correction_factor_sky[low_wl+1];
        }

        return result;        
    }
    
    // CIE XYZ and RGB versions

    public static ArHosekSkyModelState   arhosek_xyz_skymodelstate_alloc_init(
        double  turbidity, 
        double  albedo, 
        double  elevation
        )
    {
        ArHosekSkyModelState state = new ArHosekSkyModelState();
        
        state.solar_radius = TERRESTRIAL_SOLAR_RADIUS;
        state.turbidity    = turbidity;
        state.albedo       = albedo;
        state.elevation    = elevation;
        
        for(int channel = 0; channel < 3; ++channel )
        {
            ArHosekSkyModel_CookConfiguration(
                datasetsXYZ[channel], 
                state.configs[channel], 
                turbidity, 
                albedo, 
                elevation);
        
            state.radiances[channel] = 
            ArHosekSkyModel_CookRadianceConfiguration(
                datasetsXYZRad[channel],
                turbidity, 
                albedo,
                elevation);
        }
        return state;        
    }

    
    
    public static ArHosekSkyModelState   arhosek_rgb_skymodelstate_alloc_init(
        double  turbidity, 
        double  albedo, 
        double  elevation
        )
    {
        ArHosekSkyModelState state = new ArHosekSkyModelState();
    
        state.solar_radius = TERRESTRIAL_SOLAR_RADIUS;
        state.turbidity    = turbidity;
        state.albedo       = albedo;
        state.elevation    = elevation;

        for(int channel = 0; channel < 3; ++channel )
        {
            ArHosekSkyModel_CookConfiguration(
                datasetsRGB[channel], 
                state.configs[channel], 
                turbidity, 
                albedo, 
                elevation);
        
            state.radiances[channel] = 
                ArHosekSkyModel_CookRadianceConfiguration(
                datasetsRGBRad[channel],
                turbidity, 
                albedo,
                elevation);
        }
        return state;
    }
    
    public static double arhosek_tristim_skymodel_radiance(
        ArHosekSkyModelState    state,
        double                  theta,
        double                  gamma, 
        int                     channel
        )
    {        
        return ArHosekSkyModel_GetRadianceInternal(
                state.configs[channel], 
                theta, 
                gamma) 
            * state.radiances[channel];
    }
    
    static final int pieces = 45;
    static final int order = 4;

    public static double arhosekskymodel_sr_internal(
        ArHosekSkyModelState    state,
        int                     turbidity,
        int                     wl,
        double                  elevation
        )
    {
        int pos =
            (int) (pow(2.0*elevation / MATH_PI, 1.0/3.0) * pieces); // floor
    
        if ( pos > 44 ) pos = 44;
    
        final double break_x =
            pow(((double) pos / (double) pieces), 3.0) * (MATH_PI * 0.5);
        
        final double [] coefs = Arrays.copyOfRange(solarDatasets[wl], 0, (order * pieces * turbidity + order * (pos+1) - 1) + 1); //add 1, to point to the last index which is exclusive (not copied)
        int arrayIndex = coefs.length-1;
        
        double res = 0.0;
        final double x = elevation - break_x;
        double x_exp = 1.0;

        for (int i = 0; i < order; ++i)
        {
         
            res += x_exp * coefs[arrayIndex]; arrayIndex--;            
            x_exp *= x;
        }

        return  res * state.emission_correction_factor_sun[wl];
    }
    
    static double arhosekskymodel_solar_radiance_internal2(
        ArHosekSkyModelState    state,
        double                  wavelength,
        double                  elevation,
        double                  gamma
        )
    {
        assert(wavelength >= 320.0
            && wavelength <= 720.0
            && state.turbidity >= 1.0
            && state.turbidity <= 10.0);
        
        int     turb_low  = (int) state.turbidity - 1;
        double  turb_frac = state.turbidity - (double) (turb_low + 1);
    
        if ( turb_low == 9 )
        {
            turb_low  = 8;
            turb_frac = 1.0;
        }

        int    wl_low  = (int) ((wavelength - 320.0) / 40.0);
        double wl_frac = (wavelength % 40.0) / 40.0;
    
        if ( wl_low == 10 )
        {
            wl_low = 9;
            wl_frac = 1.0;
        }
        
        double direct_radiance =
            ( 1.0 - turb_frac )
          * (    (1.0 - wl_frac)
             * arhosekskymodel_sr_internal(
                     state,
                     turb_low,
                     wl_low,
                     elevation
                   )
           +   wl_frac
             * arhosekskymodel_sr_internal(
                     state,
                     turb_low,
                     wl_low+1,
                     elevation
                   )
          )
        +   turb_frac
            * (    ( 1.0 - wl_frac )
             * arhosekskymodel_sr_internal(
                     state,
                     turb_low+1,
                     wl_low,
                     elevation
                   )
           +   wl_frac
             * arhosekskymodel_sr_internal(
                     state,
                     turb_low+1,
                     wl_low+1,
                     elevation
                   )
          );
        
        double [] ldCoefficient = new double[6];
    
        for ( int i = 0; i < 6; i++ )
            ldCoefficient[i] =
              (1.0 - wl_frac) * limbDarkeningDatasets[wl_low  ][i]
            +        wl_frac  * limbDarkeningDatasets[wl_low+1][i];
    
        // sun distance to diameter ratio, squared

        final double sol_rad_sin = sin(state.solar_radius);
        final double ar2 = 1 / ( sol_rad_sin * sol_rad_sin );
        final double singamma = sin(gamma);
        double sc2 = 1.0 - ar2 * singamma * singamma;
        if (sc2 < 0.0 ) sc2 = 0.0;
        double sampleCosine = sqrt (sc2);
    
        //   The following will be improved in future versions of the model:
        //   here, we directly use fitted 5th order polynomials provided by the
        //   astronomical community for the limb darkening effect. Astronomers need
        //   such accurate fittings for their predictions. However, this sort of
        //   accuracy is not really needed for CG purposes, so an approximated
        //   dataset based on quadratic polynomials will be provided in a future
        //   release.
        
        double  darkeningFactor =
            ldCoefficient[0]
          + ldCoefficient[1] * sampleCosine
          + ldCoefficient[2] * pow( sampleCosine, 2.0 )
          + ldCoefficient[3] * pow( sampleCosine, 3.0 )
          + ldCoefficient[4] * pow( sampleCosine, 4.0 )
          + ldCoefficient[5] * pow( sampleCosine, 5.0 );
        
        direct_radiance *= darkeningFactor;

        return direct_radiance;
    }
    
    //   Delivers the complete function: sky + sun, including limb darkening.
    //   Please read the above description before using this - there are several
    //   caveats!

    public static double arhosekskymodel_solar_radiance(
        ArHosekSkyModelState         state,
        double                       theta,
        double                       gamma,
        double                       wavelength
        )
    {
        double  direct_radiance =
        arhosekskymodel_solar_radiance_internal2(
            state,
            wavelength,
            ((MATH_PI/2.0)-theta),
            gamma
            );

        double  inscattered_radiance =
        arhosekskymodel_radiance(
            state,
            theta,
            gamma,
            wavelength
            );
    
        return  direct_radiance + inscattered_radiance;
    }
    
    static float gamma(CVector3 v, CVector3 sunPosition)
    {
        return SphericalCoordinate.getRadiansBetween(v, sunPosition);
    }
    
    static float zenith(CVector3 v)
    {
        return SphericalCoordinate.thetaRadians(v);
    }
    
    public static ArHosekSkyModelState initStateRGB(double turbidity, double albedo, CVector3 sunPosition)
    {
        return arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, SphericalCoordinate.elevationRadians(sunPosition));        
    }
    
    public static ArHosekSkyModelState initStateXYZ(double turbidity, double albedo, CVector3 sunPosition)
    {
        return arhosek_xyz_skymodelstate_alloc_init(turbidity, albedo, SphericalCoordinate.elevationRadians(sunPosition));
    }
    
    public static ArHosekSkyModelState initStateRadiance(double turbidity, double albedo, CVector3 sunPosition)
    {        
        return arhosekskymodelstate_alloc_init(SphericalCoordinate.elevationRadians(sunPosition), turbidity, albedo);
    }
        
    public static boolean isInSolarDisk(CVector3 v, ArHosekSkyModelState currentState)
    {
        return SphericalCoordinate.isInsideDisk(v, (float) currentState.solarRadiusToDegrees(), (float) currentState.elevationToDegrees());            
    }
    
    public static Color getRGB(CVector3 d, ArHosekSkyModelState currentState, double exposure, double tonemapGamma)
    {
        CVector3 dir = d.copy();
                
        if(dir.y < 0.01)
        {
            return new Color();
        }       
                    
        float gamma         = gamma(dir, currentState.getSolarDirection());
        float theta         = zenith(dir);
        
        double r = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 0);
        double g = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 1);
        double b = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 2);
        
        //System.out.println(new Color(r,g,b));
        return new Color(r, g, b).mul((float)exposure).simpleGamma((float)tonemapGamma);
    }
    
    public Color getRGB_using_XYZ(CVector3 d, ArHosekSkyModelState currentState, double exposure, double tonemapGamma)
    {
        CVector3 dir = d.copy();
                
        if(dir.y < 0)
            return new Color();        
                    
        float gamma         = gamma(dir, currentState.getSolarDirection());
        float theta         = zenith(dir);
               
        double X = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 0);
        double Y = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 1);
        double Z = arhosek_tristim_skymodel_radiance(currentState, theta, gamma, 2);
        
        Color color =  RGBSpace.convertXYZtoRGB(new XYZ(X, Y, Z));   
        
        return color.mul((float)exposure).simpleGamma((float)tonemapGamma);
    }
    
    public static Color getRGB_using_radiance(CVector3 d, ArHosekSkyModelState currentState, double exposure, double tonemapGamma)
    {
        CVector3 dir = d.copy();
                
        if(dir.y < 0)
            return new Color();        
                    
        float gamma         = gamma(dir, currentState.getSolarDirection());
        float theta         = zenith(dir);
        
        float X, Y, Z;
        X = Y = Z = 0;
        
        for(int i = 320; i<=720; i+=40)
        {
            double radiance = arhosekskymodel_radiance(currentState, theta, gamma, i);
            X += CIE1931.getX(radiance, i);
            Y += CIE1931.getY(radiance, i);
            Z += CIE1931.getZ(radiance, i);
        }
        
        Color color =  RGBSpace.convertXYZtoRGB(new XYZ(X, Y, Z).mul(40));   
        return color;
    }
    
    public static Color getRGB_using_solar_radiance(CVector3 d, ArHosekSkyModelState currentState, double exposure, double tonegamma)
    {
        CVector3 dir = d.copy();
                
        if(dir.y < 0)
            return new Color();    
        
        if(!SphericalCoordinate.isInsideDisk(d, (float) currentState.solarRadiusToDegrees(), (float) currentState.elevationToDegrees()))
            return new Color();
                    
        float gamma         = gamma(dir, currentState.getSolarDirection());
        float theta         = zenith(dir);
        
        float X, Y, Z;
        X = Y = Z = 0;
        
        for(int i = 320; i<=720; i+=40)
        {
            double radiance = arhosekskymodel_solar_radiance(currentState, theta, gamma, i);
            X += CIE1931.getX(radiance, i);
            Y += CIE1931.getY(radiance, i);
            Z += CIE1931.getZ(radiance, i);
        }
        
        Color color =  RGBSpace.convertXYZtoRGB(new XYZ(X, Y, Z)).mul(0.01f);         
        return color.mul((float)exposure);
    }   
    
    public static Color getSunColor(ArHosekSkyModelState currentState, double exposure)
    {
        CVector3 v = currentState.getSolarDirection();
        return getRGB_using_solar_radiance(v, currentState, exposure, 1);
    }
    
    public static Color[] getSunSky(ArHosekSkyModelState skyState, ArHosekSkyModelState sunState, int width, int height, float exposure, float tonemap)
    {
        Color [] colors = new Color[width * height];
        
        for(int j = 0; j<height; j++)
            for(int i = 0; i<width; i++)
            {
                
                Color sun = getRGB_using_solar_radiance(SphericalCoordinate.sphericalDirection(i, j, width, height), sunState, exposure, tonemap);
                Color sky = getRGB(SphericalCoordinate.sphericalDirection(i, j, width, height), skyState, exposure, tonemap);
                float sunAlpha = Math.min(sun.getMin(), 1);

                Color col = Color.blend(sky, sun, sunAlpha);
                colors[i + j * width] = col;
            }
        return colors;
    }
    
    public static Color[] getSun(ArHosekSkyModelState sunState, int width, int height, float exposure, float tonemap)
    {
        Color [] colors = new Color[width * height];
        
        for(int j = 0; j<height; j++)
            for(int i = 0; i<width; i++)                            
                colors[i + j * width] = getRGB_using_solar_radiance(SphericalCoordinate.sphericalDirection(i, j, width, height), sunState, exposure, tonemap);                
               
        return colors;
    }
    
    public static Color[] getSky(ArHosekSkyModelState skyState, int width, int height, float exposure, float tonemap)
    {
        Color [] colors = new Color[width * height];
        
        for(int j = 0; j<height; j++)
            for(int i = 0; i<width; i++)        
                colors[i + j * width] = getRGB(SphericalCoordinate.sphericalDirection(i, j, width, height), skyState, exposure, tonemap);
                     
        return colors;
    }
    
    public static void setSunSize(double size)
    {
        TERRESTRIAL_SOLAR_RADIUS = ( ( size * DEGREES ) / 2.0 );
    }    
}
