# A Perfomrance model for hierarchical parallelism and partitioned communications.

Goal: Create a simple model that allows us to understand the compute/communication
tradeoffs when setting the hierarchical parallelism block size in a simple stencil 
code in both strong and weak scaling regimes.

Assumptions:
  1. Basic bulk synchronous model - cannot start an iteration until all of the 
     communication enabling it completes, 
  2. Early-bird and GPU-triggered communication via partitioning effectively 
     increases communication bandwidth and decreases communication latency 
     by hiding communication costs in computation.
  3. Simple postal model of communication time that needs to be modified for
     early-bird work.

Basic model structure options:
  1. Compute time + communication time
  2. Max of compute and commmunication rate over iteration (roofline model)

Model parameters - name (abbrev) - description (units):
  1. Compute intensity (i) - fixed compute per cell determined by the algorithm (time/cell)
  1. Dimension (d) - assumed to be 2 to start (dimensionless)
  1. Problem size (n) - length of one side of the mesh (cells)
  1. Cell size (c) - bytes of data per cell (bytes)
  1. Communication latency (l) - bytes of data per cell (bytes)
  1. Communication rate (r) - inverse bandwidth (sec/byte)
  1. Blocks per dimension (b) - number of blocks to divide each dimension into - we'll end 
     of with b^d total blocks (dimensionless)
  1. Compute efficiency (e) - (0,1] based on how changing hierarchical blocking impacts
     the compute time of the iteration (dimensionless). Asssume this is a (nonlinear) function 
     of (n/b) - so e(n, b)

Compute 
  * time = i*(n^d)*e(n,b) // time
  * rate = i*e(n,b) // time/cell

Communication time - assume all communication n parallel, bandwidth bounded by largest dimension
  * time
    * 2d, no overlap = l + r * c * 4 * n/b * b

When is it worth it?
  * When the amount of communication we hide is less than the reduced efficiency
