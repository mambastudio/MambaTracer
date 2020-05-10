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
import static coordinate.utility.Utility.INV_PI_F;
import static coordinate.utility.Utility.PI_F;
import static coordinate.utility.Utility.acosf;
import static coordinate.utility.Utility.atan2f;
import static coordinate.utility.Utility.cosf;
import static coordinate.utility.Utility.sinf;
import static coordinate.utility.Utility.toDegrees;
import static coordinate.utility.Utility.toRadians;
import coordinate.utility.Value2Df;
import static java.lang.Math.abs;



/**
 *
 * @author user
 */
public class SphericalCoordinate 
{
    //theta values are from +ve y-axis and covers 180 degrees to -ve y-axis
    //phi values are 360 degrees and cover a full circle horizontal surface (xz)
    //assumption here is a right hand coordinate system
    
    //http://blog.demofox.org/2013/10/12/converting-to-and-from-polar-spherical-coordinates-made-easy/
    //but here we use different axis orientation contrary to the page (see above paragraph in this class)
    
    //this class is good for sunlight coordinates and environment map 
    
    private SphericalCoordinate()
    {
        
    }
    
    public static Value2Df getRange2f(CVector3 d, float width, float height)
    {
        float u, v;
                
        //for u, atan2f has the range of 0 to 180 degrees and 0 to -180 degrees
        //therefore, dividing by inverse 2 pi gives range 0 to 0.5 and 0 to -0.5
        //but adding the result with 0.5 shift range from 0 to 1
        //https://en.wikipedia.org/wiki/Atan2
        
        u = (0.5f  + 0.5f*INV_PI_F * atan2f(d.x, -d.z));
        
        //for v, it's quite intuitive to end up having a range between 0 to 1
        v = (INV_PI_F * acosf(d.y));
        
        // scale to image space
        u *= width;
        v *= height;
       
        return new Value2Df(u, v);
    }
    
    public static int[] getRange2i(CVector3 d, float width, float height)
    {
        float u, v;
                
        //for u, atan2f has the range of 0 to 180 degrees and 0 to -180 degrees
        //therefore, dividing by inverse 2 pi gives range 0 to 0.5 and 0 to -0.5
        //but adding the result with 0.5 shift range from 0 to 1
        //https://en.wikipedia.org/wiki/Atan2
        
        u = (0.5f  + 0.5f*INV_PI_F * atan2f(d.x, -d.z));
        
        //for v, it's quite intuitive to end up having a range between 0 to 1
        v = (INV_PI_F * acosf(d.y));
        
        // scale to image space
        u *= width;
        v *= height;
       
        return new int[]{(int)u, (int)v};
    }
    
    public static float thetaRadians(CVector3 v)
    {
        return acosf(v.y);  // Y / radius -> radius = 1 since v is normalized
    }
    
    public static float thetaDegrees(CVector3 v)
    {
        return toDegrees(thetaRadians(v));
    }
    
    public static float thetaRadians(float scale)
    {
        return PI_F * scale;        //Y-axis
    }
    
    public static float thetaDegrees(float scale)
    {
        return toDegrees(thetaRadians(scale));        //Y-axis
    }
        
    public static float phiRadians(CVector3 v)
    {        
        //tantheta = sintheta/costheta -> always check whether x is sin or z is cos to remember
        return atan2f(v.x, -v.z);
    }
        
    public static float phiDegrees(CVector3 v)
    {
        return toDegrees(phiRadians(v));
    }
    
    public static float phiRadians(float scale)
    {
        return PI_F * (2 * scale - 1f);  //the PI_F deduction is to rotate it by 180 degrees to much with right hand rule coordinates
    }
    
    public static CVector3 directionRadians(float theta, float phi)
    {
        float x = sinf(phi) * sinf(theta);                
        float y = cosf(theta);
        float z = -cosf(phi) * sinf(theta); //negative coz right hand rule
        
        return new CVector3(x, y, z);
    }
    
    public static CVector3 directionDegrees(float theta, float phi)
    {
        return directionRadians(toRadians(theta), toRadians(phi));
    }
    
    public static CVector3 reverseDirectionDegrees(float theta, float phi)
    {
        return directionDegrees(theta, phi).neg();
    }
    
    public static CVector3 sphericalDirection(float i, float j, float width, float height)
    {
        //Since this is the right hand rule coordinate system, the phi covers 360 degrees
        //intuitively, but the z-axis is negative hence we subtract with PI_F to rotate
        //by 180 degrees
        
        float scalePhi = i / width;
        float scaleTheta = j / height;
        
        float phi = PI_F * (2 * scalePhi - 1f);
        float theta = scaleTheta * PI_F;
        
        return directionRadians(theta, phi);
    }
    
    public static float getRadiansBetween(CVector3 v1, CVector3 v2)
    {
        return acosf(v1.dot(v2));
    }
    
    public static float getDegreesBetween(CVector3 v1, CVector3 v2)
    {
        return toDegrees(getRadiansBetween(v1, v2));
    }
    
    
    public static float elevationRadians(CVector3 v)
    {
        float zenithDegrees = thetaDegrees(v);
        float elevationDegrees = 90 - zenithDegrees;
        return toRadians(elevationDegrees);
    }
    
    public static float elevationDegrees(CVector3 v)
    {
        float zenithDegrees = thetaDegrees(v);
        return 90 - zenithDegrees;       
                
    }
    
    public static CVector3 elevationDegrees(float elevationDegrees)
    {
        float phi = 0;
        float zenithDegrees = abs(90 - elevationDegrees); 
        
        if(elevationDegrees > 90)
        {
            phi = 180;
            
        }
        
        return directionDegrees(zenithDegrees, phi); //CONFIRM whether is zero 
    }
    
    public static CVector3 elevationRadians(float radians)
    {
        return elevationDegrees(toDegrees(radians));
    }
    
    public static boolean isInsideDisk(CVector3 v, float sizeRadiusDegrees, float elevationDegrees)
    {
        CVector3 sunDirection = elevationDegrees(elevationDegrees);
        float degreesBetween = getDegreesBetween(v, sunDirection);
        return degreesBetween <= sizeRadiusDegrees;
    }
}
