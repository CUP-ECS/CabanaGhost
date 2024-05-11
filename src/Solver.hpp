/****************************************************************************
 * Copyright (c) 2020-2022 by the CabanaGhost authors                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaGhost. CabanaGhost is                         *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANAGHOST_SOLVER_HPP
#define CABANAGHOST_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <Cabana_Grid.hpp>

#include "ProblemManager.hpp"
#include "SiloWriter.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include <mpi.h>

namespace CabanaGhost
{

//---------------------------------------------------------------------------//
namespace Approach {
  struct Flat {};
  template <std::size_t Blocks> struct Hierarchical {}; // XXX should this be number of blocks or tile size?
  struct Host {};
  struct Kernel {};
} // namespace Approach

//---------------------------------------------------------------------------//

template <unsigned long Dims, class IterationFunctor, class CompApproach, class CommApproach>
class Solver 
{
  public:
    using mesh_type = Cabana::Grid::UniformMesh<double, Dims>;
    using pm_type = ProblemManager<Dims>;
    using array_type = typename pm_type::cell_array_type;
    using view_type = typename array_type::view_type;

    template <class InitFunc>
    Solver( const std::array<int, Dims> & global_num_cells, 
            IterationFunctor& iteration_functor, const InitFunc& create_functor ) 
        : _time( 0 ), _iter_func(iteration_functor)
    {
        // Create a local grid describing our data layout
        // Create global mesh bounds.
        std::array<double, Dims> global_low_corner, global_high_corner;
        for ( int d = 0; d < Dims; ++d )
        {
            global_low_corner[d] = 0.0;
            global_high_corner[d] = global_num_cells[d];
        }
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
                               global_low_corner, global_high_corner, global_num_cells );

        // Build the mesh partitioner and global grid.
        std::array<bool, Dims> periodic;
        for (int d = 0; d < Dims; ++d) 
        {
            periodic[d] = true;
        }
        Cabana::Grid::DimBlockPartitioner<Dims> partitioner;
        auto global_grid = Cabana::Grid::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                               periodic, partitioner );
        // Build the local grid. 
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, 1 );

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<ProblemManager<Dims>>( _local_grid, create_functor );

        // Set up Silo for I/O
        _silo = std::make_unique<SiloWriter<Dims>>( *_pm );
    }

    void setup()
    {
        /* Halo the source array to get values from neighboring processes */
        _pm->gather( Version::Current() );
    }


    /* Now the various versions of code to actually compute/communicate 
     * a timestep. These are conditional on the computataional approach being
     * used. */
    void step() requires (std::same_as<Approach::Flat, CompApproach> 
                          && std::same_as<Approach::Host, CommApproach>);

    template <std::size_t Blocks, class Comp, class Comm>
        requires std::same_as<Approach::Hierarchical<Blocks>, Comp>
                 && std::same_as<Approach::Host, Comm>
    void step()
    {
        // 1. Get the data we need and then construct a functor to handle
        // parallel computation on that 
        auto local_grid = _pm->localGrid();
        auto src_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() ).view();
        auto dst_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() ).view();
        struct golFunctor gol(src_view, dst_view);

        // 2. Figure out the portion of that data that we own and need to 
        // compute. Note the assumption that the Ghost data is already up
        // to date here.
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        // We use hierarchical parallelism here to enable partitioned communication along the boundary
        // as blocks of tjhe mesh are computed. There is likely some computational cost to this.
        
        // 1. Determine the number of teams in the league (the league size), based on the block size 
        // we want to communicate in each dimension. Start assuming square blocks.
        int iextent = own_cells.extent(0), jextent = own_cells.extent(1);;
        int blocks_per_dim = 2;
        int block_size = (iextent + blocks_per_dim - 1)/blocks_per_dim;
        int league_size = blocks_per_dim * blocks_per_dim;
        int istart = own_cells.min(0), jstart = own_cells.min(1);
        int iend = own_cells.max(0), jend = own_cells.max(1);

        typedef typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type member_type;
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> mesh_policy(league_size, Kokkos::AUTO);
        Kokkos::parallel_for("Game of Life Mesh Parallel", mesh_policy, 
            KOKKOS_LAMBDA(member_type team_member) 
        {
            // Figure out the i/j pieces of the block this team member is responsible for
            int league_rank = team_member.league_rank();
            int itile = league_rank / blocks_per_dim,
                jtile = league_rank % blocks_per_dim;
            int ibase = istart + itile * block_size,
                jbase = jstart + jtile * block_size;
            int ilimit = std::min(ibase + block_size, iend),
                jlimit = std::min(jbase + block_size, jend);
            int iextent = ilimit - ibase,
                jextent = jlimit - jbase;

            // 2. Now the team of threads iterates over the block it is responsible for. Each thread
            // in the team may handle multiple indexes, depending on the size of the team.
            auto block = Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(team_member, iextent, jextent);
            Kokkos::parallel_for(block, [&](int i, int j)
            {
                gol(ibase + i, jbase + j);
            });

            // 3. Finally, any team-specific operations that need the block to be completed
            // can be done by using a team_barrier, for example block-specific communication. 
            // None is needed here since all communication is host-driven.
            // team_member.team_barrier();
        });

        // Make sure the parallel for loop is done before use its results
        Kokkos::fence();
          
        /* Halo the computed values for the next time step */
        _pm->gather( Version::Next() );

        /* Switch the source and destination arrays and advance time*/
        _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
        _time++;
    }

#if 0
    // Sketch of a hierarchical communication approach with kernel triggering, 
    // but it doesn't compile yet.
    template <std::size_t Blocks, class Comp, class Comm>
        requires std::same_as<Approach::Hierarchical<Blocks>, Comp>
                 && std::same_as<Approach::Kernel, Comm>
    void step()
    {
        // 1. Get the data we need and then construct a functor to handle
        // parallel computation on that 
        auto local_grid = _pm->localGrid();
        auto src_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                                   Version::Current() );
        auto dst_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                                   Version::Next() );
        auto src_view = src_array.view(), dst_view = dst_array.view();
        struct golFunctor gol(src_view, dst_view);
        auto halo = _pm->halo( );

        // 2. Figure out the portion of that data that we own and need to 
        // compute. Note the assumption that the Ghost data is already up
        // to date here.
        auto own_cells = 
          _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), 
                                   Cabana::Grid::Local() );

        // We use hierarchical parallelism to enable partitioned communication 
        // along the boundary as blocks of the mesh are computed. There is  
        // some computational cost versus non-hierarchical parallelism to this.
    
        // 3. Determine the number of teams in the league (the league size)
        // based on the block size we want to communicate in each dimension. 
        int iextent = own_cells.extent(0), jextent = own_cells.extent(1);;
        int iblocks = Blocks, jblocks = Blocks; 
        int iblock_size = (iextent + iblocks - 1)/iblocks,
            jblock_size = (jextent + jblocks - 1)/jblocks;
        int league_size = iblocks * jblocks;
        int istart = own_cells.min(0), jstart = own_cells.min(1);
        int iend = own_cells.max(0), jend = own_cells.max(1);

        // 4. Start the halo exchange process
	halo.gatherStart(Kokkos::DefaultExecutionSpace(), dst_array);

        // 5. Define the thread team policy which will compute elements 
        typedef typename Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type member_type;
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> mesh_policy(league_size, Kokkos::AUTO);
 
        // 6. Launch the thread teams, one per block
        Kokkos::parallel_for("Game of Life Mesh Parallel", mesh_policy, 
        	KOKKOS_LAMBDA(member_type team_member) 
        {
            // 6a. Figure out the i/j pieces of the block this team member is 
            // responsible for. 
            // XXX Should make Kokkos or Cabana helper functions to do this. XXX
	    int league_rank = team_member.league_rank();
	    int itile = league_rank / iblocks,
	        jtile = league_rank % iblocks; 
	    int ibase = istart + itile * iblock_size,
	        jbase = jstart + jtile * jblock_size;
	    int ilimit = std::min(ibase + iblock_size, iend),
	        jlimit = std::min(jbase + jblock_size, jend);
	    int iextent = ilimit - ibase,
	        jextent = jlimit - jbase;

	    // 6b. The team of threads iterates over its block. Each thread 
            // in the team may handle multiple indexes, depending on the 
            // size of the team. 
            // XXX Make a TeamThreadMDRange that takes start and end indexes 
            // in each dimension just like the standard MDRange can. XXX
	    auto block = Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, member_type>(team_member, iextent, jextent);
	    Kokkos::parallel_for(block, [&](int i, int j)
	    {
	        gol(ibase + i, jbase + j);
	    });

	    // 6c. Finally, the team is done with its block and can work on any
            // communication that the block needs. Note that this can also
            //   1. Use the thread team to pack any buffers that need to be sent
            //   2. use team_member.barrier() to synchronize before having one 
            //      team member call pready to send any data needed.
	    halo.gatherReady(Kokkos::DefaultExecutionSpace(), team_member, {itile, jtile}, dst_array);
        });

        // 7. Make sure the parallel for loop is done before use its results
        Kokkos::fence();
      
        /* 8. Finish the halo for the next time step */ 
        halo.gatherFinish( Kokkos::DefaultExecutionSpace(), dst_array );

        /* Switch the source and destination arrays and advance time*/
        _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
        _time++;
    }
#endif // if 0 for kernel triggering

    bool checkConvergence( const double tol )
        requires (std::same_as<Approach::Flat, CompApproach> 
                  && std::same_as<Approach::Host, CommApproach>)
    {
        // Compute difference between previous and currenet time step
        // grid_parallel_reduce();
        // do a global allreduce of that.
        return false;
    } 

    void solve( const int t_max, const double tol = 0.0, const int write_freq = 0)
    {
        int t = 0;
        int rank;
        bool converged = false;

        MPI_Comm_rank(_local_grid->globalGrid().comm(), &rank);

        if (write_freq > 0) {
            _silo->siloWrite( strdup( "Mesh" ), t, _time, 1 );
        }

        // Setup for the solve.
        setup();

        // Start advancing time.
        do
        {
            if ( 0 == rank )
                printf( "Step %d / %d\n", t, (int)t_max );

            step();
            t++;
            // Output mesh state periodically
            if ( write_freq && (0 == t % write_freq ))
            {
                _silo->siloWrite( strdup( "Mesh" ), t, _time, 1 );
            }
 
            if (tol > 0) {
                converged = checkConvergence(tol);
            }
        } while ( !converged && ( _time < t_max ) );
    }

  private:
    /* Solver state variables */
    int _time;
    
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
    IterationFunctor& _iter_func; // XXX Actually define this class some time
    std::unique_ptr<ProblemManager<Dims>> _pm;
    std::unique_ptr<SiloWriter<Dims>> _silo;
};

template <unsigned long Dims, class IterationFunctor,
          class CompApproach, class CommApproach>
void Solver<Dims, IterationFunctor, CompApproach, CommApproach>::step()
    requires (std::same_as<Approach::Flat, CompApproach> 
              && std::same_as<Approach::Host, CommApproach>) 
{
    // 1. Get the data we need and then construct a functor to handle
    // parallel computation on that 
    auto local_grid = _pm->localGrid();
    auto src_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                              Version::Current() ).view();
    auto dst_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                              Version::Next() ).view();

    // XXX Change this to a swap
    _iter_func.setViews(src_view, dst_view);

    // 2. Figure ouyt the portion of that data that we own and need to 
    // compute. Note the assumption that the Ghost data is already up
    // to date here.
    auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // 3. Iterate over the space of indexes we own and apply the 
    // functor in parallel to that space to calculate the 
    // output data
    Cabana::Grid::grid_parallel_for("Game of Life Mesh Parallel Loop", 
        Kokkos::DefaultExecutionSpace(), own_cells, _iter_func);

    // 4. Make sure the parallel for loop is done before use its results
    Kokkos::fence();

    // 5. Gather our ghost cells for the next time around from our
    // our neighbor's owned cells.
    _pm->gather( Version::Next() );

    /* 6. Make the state we next state the current state and advance time*/
    _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
    _time++;
}

//---------------------------------------------------------------------------//

} // end namespace CabanaGhost

#endif // end CABANAGHOST_SOLVER_HPP

