# CLTracer

A simple Java OpenCL Tracer mainly targeting OpenCL 1.2 with AMD and Intel Drivers. 

![Alt text](screenshot.png?raw=true "Title")

## Features

* BVH Skip Link
* Intuitive UI
* Memory efficient data storage (use of only Java primitives (float, int) for mesh and bvh)
* Interactive camera control

## Dependencies

* JOCL Library http://www.jocl.org/

## Future Implementation

* Fast Acceleration Structure 
  - Fast Build
    - Maximizing Parallelism in the Construction of BVHs - http://research.nvidia.com/publication/maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
  - Fast Traversal & Build
    - GPU Ray Tracing using Irregular Grids (solves teapot in a stadium problem) - https://graphics.cg.uni-saarland.de/index.php?id=939
* Scene Description
  - GL Transmission Format (glTF) as main scene format - https://github.com/KhronosGroup/glTF
  - Improve OBJ File format reading 
