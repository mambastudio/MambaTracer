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

import cl.core.env.sun.data.ArHosekSkyModelData_Spectral_Dataset.*;
import static cl.core.env.sun.data.ArHosekSkyModelData_Spectral_DatasetRad.*;
import cl.core.env.sun.data.ArHosekSkyModelData_Spectral_SolarDataset.*;

/**
 *
 * @author user
 */
public class ArHosekSkyModelData_Spectral {
    

public static final double[][] datasets =
{
	Dataset320.dataset,
	Dataset360.dataset,
	Dataset400.dataset,
	Dataset440.dataset,
	Dataset480.dataset,
	Dataset520.dataset,
	Dataset560.dataset,
	Dataset600.dataset,
	Dataset640.dataset,
	Dataset680.dataset,
        Dataset720.dataset
};

public static final double[][] datasetsRad =
{
	datasetRad320,
	datasetRad360,
	datasetRad400,
	datasetRad440,
	datasetRad480,
	datasetRad520,
	datasetRad560,
	datasetRad600,
	datasetRad640,
	datasetRad680,
        datasetRad720
};


public static final double[][] solarDatasets =
{
	SolarDataset320.solarDataset320,
	SolarDataset360.solarDataset360,
	SolarDataset400.solarDataset400,
	SolarDataset440.solarDataset440,
	SolarDataset480.solarDataset480,
	SolarDataset520.solarDataset520,
	SolarDataset560.solarDataset560,
	SolarDataset600.solarDataset600,
	SolarDataset640.solarDataset640,
	SolarDataset680.solarDataset680,
	SolarDataset720.solarDataset720
};

public static final double limbDarkeningDataset320[] =
{ 0.087657, 0.767174, 0.658123, -1.02953, 0.703297, -0.186735 };

public static final double limbDarkeningDataset360[] =
{ 0.122953, 1.01278, 0.238687, -1.12208, 1.17087, -0.424947 };

public static final double limbDarkeningDataset400[] =
{ 0.123511, 1.08444, -0.405598, 0.370629, -0.240567, 0.0674778 };

public static final double limbDarkeningDataset440[] =
{ 0.158489, 1.23346, -0.875754, 0.857812, -0.484919, 0.110895 };

public static final double limbDarkeningDataset480[] =
{ 0.198587, 1.30507, -1.25998, 1.49727, -1.04047, 0.299516 };

public static final double limbDarkeningDataset520[] =
{ 0.23695, 1.29927, -1.28034, 1.37760, -0.85054, 0.21706 };

public static final double limbDarkeningDataset560[] =
{ 0.26892, 1.34319, -1.58427, 1.91271, -1.31350, 0.37295 };

public static final double limbDarkeningDataset600[] =
{ 0.299804, 1.36718, -1.80884, 2.29294, -1.60595, 0.454874 };

public static final double limbDarkeningDataset640[] =
{ 0.33551, 1.30791, -1.79382, 2.44646, -1.89082, 0.594769 };

public static final double limbDarkeningDataset680[] =
{ 0.364007, 1.27316, -1.73824, 2.28535, -1.70203, 0.517758 };

public static final double limbDarkeningDataset720[] =
{ 0.389704, 1.2448, -1.69708, 2.14061, -1.51803, 0.440004 };

public static final double [][] limbDarkeningDatasets =
{
	limbDarkeningDataset320,
	limbDarkeningDataset360,
	limbDarkeningDataset400,
	limbDarkeningDataset440,
	limbDarkeningDataset480,
	limbDarkeningDataset520,
	limbDarkeningDataset560,
	limbDarkeningDataset600,
	limbDarkeningDataset640,
	limbDarkeningDataset680,
	limbDarkeningDataset720
};



}
