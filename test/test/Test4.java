/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

/**
 *
 * @author user
 */
public class Test4 {
    public static void main(String... args)
    {
        State state = new State();
        state.seed = (int)System.currentTimeMillis();
        
        for(int i = 0; i<10; i++)
            System.out.println(xor32(state));
    }
    
    public static float xor32(State state)
    {
        state.seed ^= state.seed << 13;
        state.seed ^= state.seed >>> 17;
        state.seed ^= state.seed << 5;
        state.seed = state.seed & Integer.MAX_VALUE; // zero out the sign bit
        return state.seed * 2.3283064365387e-10f;
    }
    
    static class State
    {
        int seed;
    }
}
