/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cl.core.data.struct;

import coordinate.struct.IntStruct;

/**
 *
 * @author user
 */
public class CNode extends IntStruct
{
    public int bound;
    public int sibling;
    public int left;
    public int right;
    public int parent;
    public int isLeaf;
    public int child;

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder(); 
        builder.append("bounds   ").append(bound).append("\n");
        builder.append("parent   ").append(parent).append("\n");
        builder.append("sibling  ").append(sibling).append("\n");
        builder.append("left     ").append(left).append(" right     ").append(right).append("\n");
        builder.append("is leaf  ").append(isLeaf).append("\n");
        builder.append("child no ").append(child).append("\n");    
        return builder.toString();
    }
}
