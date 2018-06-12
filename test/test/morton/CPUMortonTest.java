/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.morton;

import static java.lang.Integer.max;
import static java.lang.Integer.min;
import java.util.Arrays;

/**
 *
 * @author user
 * 
 * R = RadeonRays implementation
 * K = Karras implementation
 * 
 */
public class CPUMortonTest {
    public static void main(String... args)
    {
        int mortonKarras[] = new int[]{1, 2, 4, 5, 19, 24, 25, 30};
        int mortons[] = new int[] {150994944, 150994944, 153391689, 153391689, 301989888,301989888,306783378,306783378,603979776,603979776,613566756,613566756};
        //int mortonKarras[] = new int[]{1, 1, 1, 1, 1, 1, 1, 1};
        findSpanR(mortons, 0);
        findSplitR(mortons, 0, 11);
        //System.out.println(Arrays.toString(findSpanR(mortonKarras, 0)));
    }
    
    public static int[] findSpanR(int[] morton_codes, int idx)
    {
        // Find the direction of the range
        int d = Integer.signum(((delta(morton_codes, idx, idx+1) - delta(morton_codes, idx, idx-1))));
        
        // Find minimum number of bits for the break on the other side
        int delta_min = delta(morton_codes, idx, idx-d);
        
        // Search conservative far end
        int lmax = 2;
        while (delta(morton_codes, idx,idx + lmax * d) > delta_min)
            lmax *= 2;
        
        // Search back to find exact bound
        // with binary search
        int l = 0;
        int t = lmax;
        do
        {
            t /= 2;
            if(delta(morton_codes, idx, idx + (l + t)*d) > delta_min)
            {
                l = l + t;
            }
        }
        while (t > 1);
                
        // Pack span 
        int[] span = new int[2];
        span[0] = min(idx, idx + l*d);
        span[1] = max(idx, idx + l*d);
        return span;
    }
    
    public static int findSplitK(int[] sortedMortonCodes, int first, int last)
    {
        int firstCode = sortedMortonCodes[first];
        int lastCode  = sortedMortonCodes[last];

        if(firstCode == lastCode)
            return (first + last) >> 1;
        
        

        int commonPrefix = Integer.numberOfLeadingZeros(firstCode ^ lastCode);

        int split = first;
        int step = last - first;

        do
        {
            step = (step + 1) >> 1;
            int newSplit = split + step;

            if(newSplit < last)
            {
                int splitCode = sortedMortonCodes[newSplit];
                int splitPrefix = Integer.numberOfLeadingZeros(firstCode ^ splitCode);
                if(splitPrefix > commonPrefix)
                    split = newSplit;
            }
        }
        while(step > 1);

        return split;
    }
    
    public static int findSplitR(int [] sortedMortonCodes, int first, int end)
    {
        // Fetch codes for both ends
        int left = first;
        int right = end;
        
        // Calculate the number of identical bits from higher end
        int num_identical = delta(sortedMortonCodes, left, right);
        
         System.out.println(num_identical);
        
        do
        {
            // Proposed split
            int new_split = (right + left) / 2;

            // If it has more equal leading bits than left and right accept it
            if (delta(sortedMortonCodes, left, new_split) > num_identical)            
                left = new_split;            
            else            
                right = new_split;            
        }
        while (right > left + 1);
        
        return left;
    }
    
    public static int delta(int [] sortedMortonCodes, int i1, int i2)
    {
        // Select left end
        int left = min(i1, i2);
        // Select right end
        int right = max(i1, i2);
        // This is to ensure the node breaks if the index is out of bounds
        if (left < 0 || right >= sortedMortonCodes.length) 
        {            
            return -1;
        }
        // Fetch Morton codes for both ends
        int left_code = sortedMortonCodes[left];
        int right_code = sortedMortonCodes[right];

        // Special handling of duplicated codes: use their indices as a fall
        return left_code != right_code ? Integer.numberOfLeadingZeros(left_code ^ right_code) : (32 + Integer.numberOfLeadingZeros(left ^ right));
    }
    
}
