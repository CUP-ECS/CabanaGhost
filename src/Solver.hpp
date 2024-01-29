/****************************************************************************
 * Copyright (c) 2020-2022 by the CabanaGOL authors                         *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaGOL. CabanaGOL is                             *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANAGOL_SOLVER_HPP
#define CABANAGOL_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <Cabana_Grid.hpp>

#include "ProblemManager.hpp"
#include "SiloWriter.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include <mpi.h>

namespace CabanaGOL
{
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

//---------------------------------------------------------------------------//
namespace Approach {
  struct FlatHost {};
  template <std::size_t Blocks> struct HierarchicalHost {};
  template <std::size_t Blocks> struct HierarchicalKernel {};
} // namespace Approach

//---------------------------------------------------------------------------//

template <class ExecutionSpace, class MemorySpace, int Dims, class CompApproach>
class Solver : public SolverBase
{
  public:
    using mesh_type = Cabana::Grid::UniformMesh<double, Dims>;

    template <class InitFunc>
    Solver( const std::array<int, Dims> & global_num_cells, 
            const InitFunc& create_functor ) 
        : _time( 0.0 )
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
        _pm = std::make_unique<ProblemManager<ExecutionSpace, MemorySpace>>( _local_grid, create_functor );

        // Set up Silo for I/O
        _silo = std::make_unique<SiloWriter<ExecutionSpace, MemorySpace>>( *_pm );
    }

    void setup()
    {
        /* Halo the source array to get values from neighboring processes */
        _pm->gather( Version::Current() );
    }

    struct golFunctor {
        using view_type = Kokkos::View<double ***, MemorySpace>;
        view_type _src_array, _dst_array;

        KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
            double sum = 0.0;
            int ii, jj;
            for (ii = -1; ii <= 1; ii++) {
                for (jj = -1; jj <= 1; jj++) {
                    if ((ii == 0) && (jj == 0)) continue;
                    sum += _src_array(i + ii,j + jj, 0);
                }
            }
            if (_src_array(i, j, 0) == 0.0) {
                if ((sum > 2.99) && (sum < 3.01))
                    _dst_array(i, j, 0) = 1.0;
                else 
                    _dst_array(i, j, 0) = 0.0;
            } else {
                if ((sum >= 1.99) && (sum <= 3.01))
                    _dst_array(i, j, 0) = 1.0;
                else 
                    _dst_array(i, j, 0) = 0.0;
            }
        };
        golFunctor(view_type s, view_type d)
            : _src_array(s), _dst_array(d)
        {}
    };

    /* Now the various versions of code to actually compute/communicate 
     * a timestep. These are conditional on the computataional approach being
     * used. This currently uses a clever but messy C++14 construct - 
     * enable_if_t and SFINAE - and should be replaced with C++20 requires
     * onece is is more broadly available. */
    template <>
        requires std::is_same<Approach::FlatHost, CompApproach>
    void step() 
    {
        // 1. Get the data we need and then construct a functor to handle
        // parallel computation on that 
        auto local_grid = _pm->localGrid();
        auto src_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );
        auto dst_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() );
        struct golFunctor gol(src_array, dst_array);

        // 2. Figure ouyt the portion of that data that we own and need to 
        // compute. Note the assumption that the Ghost data is already up
        // to date here.
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        // 3. Iterate over the space of indexes we own and apply the 
        // functor in parallel to that space to calculate the 
        // output data
        Cabana::Grid::grid_parallel_for("Game of Life Mesh Parallel Loop", 
            ExecutionSpace(), own_cells, gol);

        // 4. Make sure the parallel for loop is done before use its results
        Kokkos::fence();
 
        // 5. Gather our ghost cells for the next time around from our
        // our neighbor's owned cells.
        _pm->gather( Version::Next() );

        /* 6. Make the state we next state the current state and advance time*/
        _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
        _time++;
    }

    template <std::size_t Blocks>
        requires std::is_same<Approach::HierarchicalHost<Blocks>, CompApproach>
    void step()
    {
        // 1. Get the data we need and then construct a functor to handle
        // parallel computation on that 
        auto local_grid = _pm->localGrid();
        auto src_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );
        auto dst_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() );
        struct golFunctor gol(src_array, dst_array);

        // 2. Figure ouyt the portion of that data that we own and need to 
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

        typedef typename Kokkos::TeamPolicy<ExecutionSpace>::member_type member_type;
        Kokkos::TeamPolicy<ExecutionSpace> mesh_policy(league_size, Kokkos::AUTO);
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
            team_member.team_barrier();
        });

        // Make sure the parallel for loop is done before use its results
        Kokkos::fence();
          
        /* Halo the computed values for the next time step */
        _pm->gather( Version::Next() );

        /* Switch the source and destination arrays and advance time*/
        _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
        _time++;
    }

    template <std::size_t Blocks>
        requires std::is_same<Approach::HierarchicalKernel<Blocks>, CompApproach>
    void step()
    {
        // 1. Get the data we need and then construct a functor to handle
        // parallel computation on that 
        auto local_grid = _pm->localGrid();
        auto src_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );
        auto dst_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() );
        struct golFunctor gol(src_array, dst_array);

        // 2. Figure ouyt the portion of that data that we own and need to 
        // compute. Note the assumption that the Ghost data is already up
        // to date here.
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        // We use hierarchical parallelism here to enable partitioned communication 
        // along the boundary as blocks of the mesh are computed. There is likely 
        // some computational cost versus pure non-hierarchical parallelism to this.
    
        // 3. Determine the number of teams in the league (the league size), based 
        // on the block size we want to communicate in each dimension. 
        int iextent = own_cells.extent(0), jextent = own_cells.extent(1);;
        int iblocks = Blocks, jblocks = Blocks; 
        int iblock_size = (iextent + iblocks - 1)/iblocks,
            jblock_size = (jextent + jblocks - 1)/jblocks;
        int league_size = iblocks * jblocks;
        int istart = own_cells.min(0), jstart = own_cells.min(1);
        int iend = own_cells.max(0), jend = own_cells.max(1);

        // 4. Start the halo exchange process
	_pm->gatherStart(Version::Next());

        // 5. Define the thread team policy which will compute elements 
        typedef typename Kokkos::TeamPolicy<ExecutionSpace>::member_type member_type;
        Kokkos::TeamPolicy<ExecutionSpace> mesh_policy(league_size, Kokkos::AUTO);
 
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

	    // 6b. The team of threads iterates over the block it is responsible for.
            // Each thread in the team may handle multiple indexes, depending on the 
            // size of the team. 
            // XXX We should make a TeamThreadMDRange that takes start and end indexes 
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
            //  XXX Capture of this is a problem here - capture ta halo instead! XXX
	    _pm->gatherReady(block, itile, jtile); 
	    // if (onLeftBoundary() ) {
	    //    Kokkos::parallel_for() {
	    //	   pack our parts of the left boundary 
            //    } 
            //}
	   // team_member.barrier();
           // if (onLeftBoundary() && team_member.rank() == 0) MPI_Pready(leftbuffer, partitionnum); 

        });

        // 7. Make sure the parallel for loop is done before use its results
        // XXX Is/should this be necessary?
        Kokkos::fence();
      
        /* 8. Finish the halo for the next time step - this probably has to unpack 
         * since we don't have a kernel running any longer to do that. 
         * XXX Alternatives XXX:
         *     1. Have the gather code in the parallel loop above to do that 
         *        with pArrived, which may open a synchronization can of worms
         *        since kernel code may not be preemptible. 
         *     2. Have this code synchronize with other kernels already queued up
         *        for unpacking on a seperate stream once buffers have come in.
         *        *could* set up those kernels (or Kokkos tasks...) Think about the
         *        right thing to do here! */
        _pm->gatherFinish( Version::Next() );

        /* Switch the source and destination arrays and advance time*/
        _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
        _time++;
    }

    void solve( const double t_final, const int write_freq ) override
    {
        int t = 0;
        int rank;

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
                printf( "Step %d / %d\n", t, (int)t_final );

            step();
            t++;
            // Output mesh state periodically
            if ( write_freq && (0 == t % write_freq ))
            {
                _silo->siloWrite( strdup( "Mesh" ), t, _time, 1 );
            }
        } while ( ( _time < t_final ) );
    }

  private:
    /* Solver state variables */
    int _time;
    
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;

    std::unique_ptr<ProblemManager<ExecutionSpace, MemorySpace>> _pm;
    std::unique_ptr<SiloWriter<ExecutionSpace, MemorySpace>> _silo;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc>
std::unique_ptr<SolverBase>
createSolver( const std::string& device,
              const std::array<int, 2>& global_num_cell,
              const InitFunc& create_functor )
{
    if ( 0 == device.compare( "serial" ) )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_unique<
            Solver<Kokkos::Serial, Kokkos::HostSpace, 2, Approach::FlatHost>
            >(global_num_cell, create_functor);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "threads" ) )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        return std::make_unique<
            Solver<Kokkos::Threads, Kokkos::HostSpace, 2, Approach::FlatHost>
            >( global_num_cell, create_functor );
#else
        throw std::runtime_error( "Threads Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_unique
            Solver<Kokkos::OpenMP, Kokkos::HostSpace, 2>>(global_num_cell, create_functor);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#if defined(KOKKOS_ENABLE_CUDA)
        return std::make_unique<
            Solver<Kokkos::Cuda, Kokkos::CudaSpace, 2, Approach::FlatHost>>(global_num_cell, create_functor);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_unique<Solver<Kokkos::Experimental::HIP, 
            Kokkos::Experimental::HIPSpace, 2>>(global_bounding_box, create_functor);
#else
        throw std::runtime_error( "HIP Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end CABANAGOL_SOLVER_HPP

