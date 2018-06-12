#define LEAFIDX(i) (i)
#define NODEIDX(i) (l_size + i)

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


__kernel void test(global BVHNode* nodes, global BoundingBox* bounds, global int* nodeSize, global int* leafSize)
{
      int n_size = nodeSize[0]; int l_size = leafSize[0]; int t_size = n_size + l_size;
      for(int i = 0; i<t_size; i++)
      {
          nodes[i].left = 3;
          nodes[i].parent = 30;
      }
}
