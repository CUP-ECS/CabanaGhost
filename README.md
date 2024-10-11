# CabanaGhost - Trivial Bulk Synchronous Computation/Communication on a Regular Mesh

This is an implementation of the multi-dimensional regular mesh iteration and 
halo exhange in Cabana/Kokkos, starting with 2D Game of Life. It is meant as 
a bulk synchonous parallel regular mesh benchmark for both teaching modern 
parallel programming and exploring programming and communication tradeoffs
in post-ECP performance portability frameworks. The initial research target is 
examining potential interactions between halo communication primitives and 
different forms of Kokkos parallelism, particularly hierarchical parallelism,

## Getting Started
To clone this directory for use with the the included setup script, utilize:

`git clone -b blt --recurse-submodules -j16 git@github.com:CUP-ECS/CabanaGhost.git $HOME/repos/CabanaGhost`