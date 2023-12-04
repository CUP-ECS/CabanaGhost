# CabanaGOL - Trivial Bulk Synchronous Computation on a Regular Mesh

This is an implementation of the Game of Life in Cabana/Kokkos. It is meant to be 
basically a bulk synchonous parallel regular mesh benchmark for both teaching
modern parallel programming and exploring programming and communication tradeoffs
in post-ECP performance portability frameworks. The initial research target is 
examining potential interactions between communication primitives and hierarchical 
Kokkos parallelism,
