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
    virtual void setup( void ) = 0;
    virtual void step( void ) = 0;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

//---------------------------------------------------------------------------//

template <class ExecutionSpace, class MemorySpace>
class Solver : public SolverBase
{
  public:
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    template <class InitFunc>
    Solver( const std::array<int, 2> & global_num_cells, 
            const InitFunc& create_functor ) 
        : _time( 0.0 )
    {
        // Create a local grid describing our data layout
        // Create global mesh bounds.
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        {
            global_low_corner[d] = 0.0;
            global_high_corner[d] = global_num_cells[d];
        }
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
                               global_low_corner, global_high_corner, global_num_cells );

        // Build the mesh partitioner and global grid.
        std::array<bool, 2> periodic = {true, true};
        Cabana::Grid::DimBlockPartitioner<2> partitioner;
        auto global_grid = Cabana::Grid::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                               periodic, partitioner );
        // Build the local grid. 
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, 1 );

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<ProblemManager<ExecutionSpace, MemorySpace>>( _local_grid, create_functor );

        // Set up Silo for I/O
        _silo = std::make_unique<SiloWriter<ExecutionSpace, MemorySpace>>( *_pm );
    }

    void setup() override
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

    /* This code assumes the halo for the current step is already done, and we have to do the
     * the halo for the next communication step. */
    void step() override
    {
        auto local_grid = _pm->localGrid();
        auto src_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );
        auto dst_array = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() );
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        struct golFunctor gol(src_array, dst_array);

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

            // 3. Finally, the team is done with its block and can barrier and have one thread
            // in the team signal any communication that needs to be done
            team_member.team_barrier();
        });
          
        /* Halo the computed values for the next time step */
        _pm->gather( Version::Next() );

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
            Solver<Kokkos::Serial, Kokkos::HostSpace>>(global_num_cell, create_functor);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "threads" ) )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        return std::make_unique<
            Solver<Kokkos::Threads, Kokkos::HostSpace>>( global_num_cell, create_functor );
#else
        throw std::runtime_error( "Threads Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_unique
            Solver<Kokkos::OpenMP, Kokkos::HostSpace>>(global_num_cell, create_functor);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#if defined(KOKKOS_ENABLE_CUDA)
        return std::make_unique<
            Solver<Kokkos::Cuda, Kokkos::CudaSpace>>(global_num_cell, create_functor);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_unique<Solver<Kokkos::Experimental::HIP, 
            Kokkos::Experimental::HIPSpacer>>(global_bounding_box, create_functor);
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

