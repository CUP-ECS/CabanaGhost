/****************************************************************************
 * Copyright (c) 2021 by the CabanaGOL authors                              *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabanaGOL benchmark. CabanaGOL is               *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANAGOL_MESH_HP
#define CABANAGOL_MESH_HP

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace CabanaGOL
{
//---------------------------------------------------------------------------//
/*!
  \class Mesh
  \brief Logically and spatially uniform Cartesian mesh.
*/
template <int Dim, class ExecutionSpace, class MemorySpace>
class Mesh
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using mesh_type = Cabana::Grid::UniformMesh<double, Dim>;

    // Construct a mesh.
    Mesh( const Kokkos::Array<double, 2 * Dim>& global_bounding_box,
          const std::array<int, Dim>& global_num_cell,
          const Cabana::Grid::BlockPartitioner<Dim>& partitioner,
          const int halo_cell_width, MPI_Comm comm )
    {
        // Make a copy of the global number of cells so we can modify it.
        std::array<int, Dim> num_cell = global_num_cell;

        // Compute the cell size.
        double cell_size =
            ( global_bounding_box[Dim] - global_bounding_box[0] ) / num_cell[0];

        // Because the mesh is uniform width in all directions, check that the
        // domain is evenly divisible by the cell size in each dimension
        // within round-off error.
        for ( int d = 0; d < Dim; ++d )
        {
            double extent = num_cell[d] * cell_size;
            if ( std::abs( extent - ( global_bounding_box[d + Dim] -
                                      global_bounding_box[d] ) ) >
                 double( 10.0 ) * std::numeric_limits<double>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double, Dim> global_low_corner, global_high_corner;
        for ( int d = 0; d < Dim; ++d )
        {
            global_low_corner[d] = global_bounding_box[d];
            global_high_corner[d] = global_bounding_box[d + Dim];
        }

        for ( int d = 0; d < Dim; ++d )
        {
            _min_domain_global_cell_index[d] = 0;
            _max_domain_global_cell_index[d] = num_cell[d] - 1;
        }

        // Create the global mesh.
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, num_cell );

        // Build the global grid.
        std::array<bool, Dim> periodic;
        for ( int i = 0; i < Dim; i++ )
            periodic[i] = false;

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );

        // Build the local grid.
        int halo_width = halo_cell_width;
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        // Build the local mesh. XXX Why is this hard to share? Is it expensive?
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *_local_grid );
        _local_mesh =
            std::make_shared<Cabana::Grid::LocalMesh<memory_space, mesh_type>>(
                local_mesh );

        MPI_Comm_rank( comm, &_rank );
    }

    // Get the local grid.
    const std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>>& localGrid() const
    {
        return _local_grid;
    }

    // Get the local mesh.
    const std::shared_ptr<Cabana::Grid::LocalMesh<memory_space, mesh_type>>&
    localMesh() const
    {
        return _local_mesh;
    }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().cellSize( 0 );
    }

    // Get the minimum node index in the domain.
    Kokkos::Array<int, Dim> minDomainGlobalCellIndex() const
    {
        return _min_domain_global_cell_index;
    }

    // Get the maximum node index in the domain.
    Kokkos::Array<int, Dim> maxDomainGlobalCellIndex() const
    {
        return _max_domain_global_cell_index;
    }

    int rank() const { return _rank; }

  public:
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
    std::shared_ptr<Cabana::Grid::LocalMesh<memory_space, mesh_type>> _local_mesh;

    Kokkos::Array<int, Dim> _min_domain_global_cell_index;
    Kokkos::Array<int, Dim> _max_domain_global_cell_index;
    int _rank;
};

//---------------------------------------------------------------------------//

} // end namespace CabanaGOL

#endif // end CABANAGOL_MESH_HP
