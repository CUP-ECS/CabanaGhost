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
    }

    void step() override
    {
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
    
    std::shared_ptr<Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<double, 2>>> _local_grid;
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

