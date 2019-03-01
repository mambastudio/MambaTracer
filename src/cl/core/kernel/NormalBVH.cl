typedef struct
{
   int bound;
   int sibling;
   int left;
   int right;
   int parent;
   int isLeaf;
   int child;

}BVHNode;

bool intersectGlobal(global Ray* ray, global Intersection* isect, TriangleMesh mesh, global BVHNode* nodes, global BoundingBox* bounds)
{
     //BVH Accelerator
     bool hit = false;

     int nodeId = 0;
     long bitstack = 0;                      //be careful when you use a 32 bit integer. For deeper hierarchy traversal may lead into an infinite loop for certain scenes, hence 64 bit long is preferable
     int parentId = 0, siblingId = 0;

     for(;;)
     {
        while(!nodes[nodeId].isLeaf)
        { 
            BVHNode node        = nodes[nodeId];
            parentId            = node.parent;
            siblingId           = node.sibling;
                  
            BVHNode left        = nodes[node.left];
            BVHNode right       = nodes[node.right];
            
            float leftT[2];
            float rightT[2];
            bool leftHit        = intersectBoundT(*ray, bounds[left.bound], leftT);
            bool rightHit       = intersectBoundT(*ray, bounds[right.bound], rightT);

            if(!leftHit && !rightHit)
                 break;
              
            bitstack <<= 1; //push 0 bit into bitstack to skip the sibling later
                  
            if(leftHit && rightHit)
            {                    
                nodeId = (rightT[0] < leftT[0]) ? node.right : node.left;
                bitstack |= 1; //change skip code to 1 to traverse the sibling later
            }
            else
            {
                nodeId = leftHit ? node.left : node.right;                   
            }
        }

        if(nodes[nodeId].isLeaf)
        {
            BVHNode node        = nodes[nodeId];
            if(intersectTriangleGlobal(ray, isect, mesh, node.child))
                hit = true;

            //This is not in the paper.
            parentId            = node.parent;
            siblingId           = node.sibling;
        }
        
        while ((bitstack & 1) == 0)  //while skip bit in the top stack is 0 traverse up the tree
        {
            if (bitstack == 0) return hit;  //if bitstack is 0 meaning stack is empty, it is now safe to exit the tree and return hit
            nodeId = parentId;
            BVHNode node = nodes[nodeId];
            parentId = node.parent;
            siblingId = node.sibling;
            bitstack >>= 1;               //pop the bit in top most part os stack by right bit shifting
        }
        nodeId = siblingId;
        bitstack ^= 1;
     }

}      