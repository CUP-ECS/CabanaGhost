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
  struct Stream {};
  struct Kernel {};
} // namespace Approach

//---------------------------------------------------------------------------//

// A base class to use for the solver that abstracts away the template arguments and 
// in particular the specific communication backend being used.
class SolverBase
{
  public: 
    virtual ~SolverBase() = default;
    virtual void solve( const int t_max, const double tol = 0.0, const int write_freq = 0) = 0;
};

template <class ExecutionSpace, class CommunicationSpace, unsigned long Dims, class IterationFunctor, class CompApproach, class CommApproach>
class Solver : public SolverBase
{
  public:
    using mesh_type = Cabana::Grid::UniformMesh<double, Dims>;
    using execution_space = ExecutionSpace;
    using communication_space = CommunicationSpace;
    using pm_type = ProblemManager<ExecutionSpace, CommunicationSpace, Dims>;
    using array_type = typename pm_type::cell_array_type;
    using view_type = typename array_type::view_type;

    template <class InitFunc>
    Solver( const std::array<int, Dims> & global_num_cells, bool periodic, 
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
        std::array<bool, Dims> p;
        for (int d = 0; d < Dims; ++d) 
        {
            p[d] = periodic;
        }
        Cabana::Grid::DimBlockPartitioner<Dims> partitioner;
        auto global_grid = Cabana::Grid::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                               p, partitioner );
        // Build the local grid. 
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, 1 );

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<ProblemManager<execution_space, communication_space, Dims>>( _local_grid, create_functor );

        // Set up Silo for I/O
        _silo = std::make_unique<SiloWriter<pm_type, Dims>>( *_pm );
    }

    void setup()
    {
        /* Halo the source array to get values from neighboring processes */
        _pm->enqueueGather( Version::Current() );
    }


    /* Now the various versions of code to actually compute/communicate 
     * a timestep. These are conditional on the computational approach being
     * used. */
    void step() requires (std::same_as<Approach::Flat, CompApproach> 
                          && (std::same_as<Approach::Host, CommApproach>
                              || std::same_as<Approach::Stream, CommApproach>));

    template <std::size_t Blocks>
    void step() requires (std::same_as<Approach::Hierarchical<Blocks>, CompApproach>
                          && (std::same_as<Approach::Host, CommApproach>
                              || std::same_as<Approach::Stream, CommApproach>));

    struct MaxDifferenceFunctor
    {
        view_type _v1, _v2;

        KOKKOS_INLINE_FUNCTION void
        operator() (const int i, const int j, const int k, double& max) const
            requires (Dims == 3)
        {
            double diff = _v2(i, j, k, 0) - _v1(i, j, k, 0);
            diff = diff < 0.0 ? -diff : diff;
            if (diff > max) max = diff;
        }

        KOKKOS_INLINE_FUNCTION void
        operator() (const int i, const int j, double& max) const
            requires (Dims == 2)
        {
            double diff = _v2(i, j, 0) - _v1(i, j, 0);
            diff = diff < 0.0 ? -diff : diff;
            if (diff > max) max = diff;
        }

        MaxDifferenceFunctor(view_type v1, view_type v2)
            : _v1(v1), _v2(v2)
        {}
    };

    bool checkConvergence( const double tol )
    {
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        auto src_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                                  Version::Current() ).view();
        auto dst_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                                  Version::Next() ).view();
        auto exec_space = execution_space();

        // Iterate over the space of indexes we own and apply the 
        // functor in parallel to that space to calculate the 
        // output data
        MaxDifferenceFunctor mdf(src_view, dst_view);
        double max = 0;
        Cabana::Grid::grid_parallel_reduce("CabanaGhost Convergence Reduction",
            exec_space, own_cells, mdf, max);

        exec_space.fence();

        // XXX We need to figure out an interface for this that is generalizable.
        MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        return max < tol;
    } 

    virtual void solve( const int t_max, const double tol = 0.0, const int write_freq = 0) override
    {
        int t = 0;
        int rank;
        bool converged = false;
        auto exec_space = execution_space();
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
                if constexpr (std::same_as<Approach::Stream, CommApproach>) {
                    exec_space.fence(); // If we're doing I/O, we need to fence the 
                                        // stream so that the data is up to date.
                }
                _silo->siloWrite( strdup( "Mesh" ), t, _time, 1 );
            }
 
            if (tol > 0.0) {
                converged = checkConvergence(tol);
            }
        } while ( !converged && ( _time < t_max ) );
        exec_space.fence(); // In case everything was able to be queued to stream.
    }

  private:
    /* Solver state variables */
    int _time;
    
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
    IterationFunctor& _iter_func; // XXX Actually define this class some time
    std::unique_ptr<ProblemManager<execution_space, communication_space, Dims>> _pm;
    std::unique_ptr<SiloWriter<pm_type, Dims>> _silo;
};

template <class ExecutionSpace, class CommunicationSpace, 
          unsigned long Dims, class IterationFunctor,
          class CompApproach, class CommApproach>
void Solver<ExecutionSpace, CommunicationSpace, Dims, IterationFunctor, CompApproach, CommApproach>::step()
    requires (std::same_as<Approach::Flat, CompApproach> 
              && (std::same_as<Approach::Host, CommApproach> 
                  || std::same_as<Approach::Stream, CommApproach>))
{
    // 1. Get the data we need and then construct a functor to handle
    // parallel computation on that 
    auto local_grid = _pm->localGrid();
    auto src_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                              Version::Current() ).view();
    auto dst_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), 
                              Version::Next() ).view();
    auto exec_space = execution_space();

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
        exec_space, own_cells, _iter_func);

    // 4. Gather our ghost cells for the next time around from our
    // our neighbor's owned cells.
    if constexpr (std::same_as<Approach::Host, CommApproach>) {
        exec_space.fence();
        _pm->gather( Version::Next() );
    } else {
        _pm->enqueueGather( Version::Next() );
    }

    /* 5. Make the state we next state the current state and advance time*/
    _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
    _time++;
}

template <class ExecutionSpace, class CommunicationSpace,
          unsigned long Dims, class IterationFunctor,
          class CompApproach, class CommApproach>
template <std::size_t Blocks>
void Solver<ExecutionSpace, CommunicationSpace, Dims, IterationFunctor, CompApproach, CommApproach>::step()
  requires (std::same_as<Approach::Hierarchical<Blocks>, CompApproach>
                 && (std::same_as<Approach::Host, CommApproach> 
                     || std::same_as<Approach::Stream, CommApproach>))
{
    // 1. Get the data we need and then construct a functor to handle
    // parallel computation on that 
    auto local_grid = _pm->localGrid();
    auto src_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() ).view();
    auto dst_view = _pm->get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() ).view();

    // 2. Figure out the portion of that data that we own and need to 
    // compute. Note the assumption that the Ghost data is already up
    // to date here.
    auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // We use hierarchical parallelism here to enable partitioned communication along the boundary
    // as blocks of tjhe mesh are computed. There is likely some computational cost to this.
    
    // 1. Determine the number of teams in the league (the league size), based on the block size 
    // we want to communicate in each dimension. Start assuming square blocks.
    int iextent = own_cells.extent(0), jextent = own_cells.extent(1);;
    int blocks_per_dim = Blocks;
    int block_size = (iextent + blocks_per_dim - 1)/blocks_per_dim;
    int league_size = blocks_per_dim * blocks_per_dim;
    int istart = own_cells.min(0), jstart = own_cells.min(1);
    int iend = own_cells.max(0), jend = own_cells.max(1);
    auto f = _iter_func;
    auto exec_space = execution_space();

    typedef typename Kokkos::TeamPolicy<execution_space>::member_type member_type;
    Kokkos::TeamPolicy<execution_space> mesh_policy(league_size, Kokkos::AUTO);
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
            f(ibase + i, jbase + j);
        });

        // 3. Finally, any team-specific operations that need the block to be completed
        // can be done by using a team_barrier, for example block-specific communication. 
        // None is needed here since all communication is host or stream driven.
        // team_member.team_barrier();
    });

    // 4. Gather our ghost cells for the next time around from our
    // our neighbor's owned cells.
    if constexpr (std::same_as<Approach::Host, CommApproach>) {
        exec_space.fence();
        _pm->gather( Version::Next() );
    } else {
        _pm->enqueueGather( Version::Next() );
    }

    /* 5. Switch the source and destination arrays and advance time*/
    _pm->advance(Cabana::Grid::Cell(), Field::Liveness());
    _time++;
}

template <class ExecutionSpace, int Dims, class CompApproach, class CommApproach, 
          class IterationFunc, class InitFunc>
std::shared_ptr<SolverBase>
createHaloSolver( std::array<int, Dims> global_num_cells, bool periodic, 
		  std::string comm_backend, IterationFunc halo, InitFunc initializer)
{
    if (comm_backend.compare("mpi") == 0) {
        return std::make_shared<
            Solver<ExecutionSpace, Cabana::CommSpace::Mpi, Dims, IterationFunc,
		CompApproach, CommApproach>>(
                global_num_cells, periodic, halo, initializer);
    } else if (comm_backend.compare("mpi-advance") == 0) {
#ifdef Cabana_ENABLE_MPI_ADVANCE
        return std::make_shared<
            Solver<ExecutionSpace, Cabana::CommSpace::MpiAdvance, Dims, IterationFunc,
		CompApproach, CommApproach>>(
                    global_num_cells, periodic, halo, initializer);
#else
        throw std::runtime_error( "MPI Advance Backend Not Enabled" );
#endif
    } else if (comm_backend.compare("mpich") == 0) {
#ifdef Cabana_ENABLE_MPICH
        return std::make_shared<
            Solver<ExecutionSpace, Cabana::CommSpace::Mpich, Dims, IterationFunc,
		CompApproach, CommApproach>>(
                global_num_cells, periodic, halo, initializer);
#else
        throw std::runtime_error( "MPICH Backend Not Enabled" );
#endif
    } else if (comm_backend.compare("cray-mpi") == 0) {
#ifdef Cabana_ENABLE_MPICH
        return std::make_shared<
            Solver<ExecutionSpace, Cabana::CommSpace::CrayMpi, Dims, IterationFunc,
		CompApproach, CommApproach>>(
                global_num_cells, periodic, halo, initializer);
#else
        throw std::runtime_error( "Cray MPI Backend Not Enabled" );
#endif
    } else {
        throw std::runtime_error("invalid communication backed");
        return nullptr;
    }
} 
//---------------------------------------------------------------------------//

} // end namespace CabanaGhost

#endif // end CABANAGHOST_SOLVER_HPP

