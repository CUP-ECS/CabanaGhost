/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Problem manager class that stores the mesh and the state data and performs
 * scatters and gathers
 */

#ifndef CABANAGOL_PROBLEMMANAGER_HPP
#define CABANAGOL_PROBLEMMANAGER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace CabanaGOL
{

/**
 * @namespace Field
 * @brief Version namespace to track whether the current or next array version
 *is requested
 **/
namespace Version
{

/**
 * @struct Current
 * @brief Tag structure for the current values of field variables. Used when
 * values are only being read or the algorithm allows the variable to be
 *modified in place
 **/
struct Current
{
};

/**
 * @struct Next
 * @brief Tag structure for the values of field variables at the next timestep.
 * Used when values being written cannot be modified in place. Note that next
 * values are only written, current values are read or written.
 **/
struct Next
{
};

} // namespace Version

/**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
namespace Field
{

/**
 * @struct Liveness
 * @brief Tag structure for the liveness state of a particular cell
 **/
struct Liveness
{
};

}; // end namespace Field

/**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and state values, and
 * to perform gathers and scatters in the approprate number of dimensions.
 **/
template <class ExecutionSpace, class MemorySpace>
class ProblemManager
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using cell_array = Cabana::Grid::Array<double, Cabana::Grid::Cell, mesh_type, 
                           MemorySpace>;
    using grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    template <class InitFunc>
    ProblemManager( const std::shared_ptr<grid_type>& local_grid,
                    const InitFunc& create_functor )
        : _local_grid( local_grid )
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associuated with them?
        auto cell_scalar_layout =
            Cabana::Grid::createArrayLayout( _local_grid, 1, Cabana::Grid::Cell() );

        // The actual arrays storing mesh quantities
        _liveness_curr = Cabana::Grid::createArray<double, MemorySpace>(
            "liveness", cell_scalar_layout );
        _liveness_next = Cabana::Grid::createArray<double, MemorySpace>(
            "liveness", cell_scalar_layout );
        Cabana::Grid::ArrayOp::assign( *_liveness_curr, 0.0, Cabana::Grid::Ghost() );
        Cabana::Grid::ArrayOp::assign( *_liveness_next, 0.0, Cabana::Grid::Ghost() );

        // Halo patterns for the just liveness. This halo is just one cell deep,
        // as we only look at that much data to calculate changes in state.
        int halo_depth = _local_grid->haloCellWidth();
        _halo = Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<2>(), 
                                          halo_depth, *_liveness_curr );

        // Initialize State Values ( liveness )
        initialize( create_functor );
    }

    /**
     * Initializes state values in the cells
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize( const InitFunctor& create_functor )
    {
        // Get State Arrays
        auto l = get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );

        // Loop Over All Owned Cells ( i, j )
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(),
                                                  Cabana::Grid::Local() );
        int index[2] = { 0, 0 };
        Kokkos::parallel_for(
            "Initialize Cells`",
            Cabana::Grid::createExecutionPolicy( own_cells, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Get Coordinates Associated with Indices ( i, j )
                int coords[2] = { i, j };
                create_functor( Cabana::Grid::Cell(), Field::Liveness(), coords,
                                l( i, j, 0 ) );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<grid_type> localGrid() const
    {
        return _local_grid;
    };

    /**
     * Return Liveness Field
     * @param Location::Cell
     * @param Field::Liveness
     * @param Version::Current
     * @return Returns view of current liveness at cell centers
     **/
    typename cell_array::view_type get( Cabana::Grid::Cell, Field::Liveness,
                                        Version::Current ) const
    {
        return _liveness_curr->view();
    };

    /**
     * Return Liveness Field
     * @param Location::Cell
     * @param Field::Liveness
     * @param Version::Next
     * @return Returns view of next liveness at cell centers
     **/
    typename cell_array::view_type get( Cabana::Grid::Cell, Field::Liveness,
                                        Version::Next ) const
    {
        return _liveness_next->view();
    };

    /**
     * Make the next version of a field the current one
     * @param Cabana::Grid::Cell
     * @param Field::Liveness
     **/
    void advance( Cabana::Grid::Cell, Field::Liveness )
    {
        _liveness_curr.swap( _liveness_next );
    }

    /**
     * Standard one-deep halo pattern for mesh fields
     */
    std::shared_ptr<halo_type> halo() const
    {
        return _halo;
    }

    /**
     * Gather State Data from Neighbors
     * @param Version
     **/
    void gather( Version::Current ) const
    {
        _halo->gather( ExecutionSpace(), *_liveness_curr );
    };
    void gather( Version::Next ) const
    {
        _halo->gather( ExecutionSpace(), *_liveness_next );
    };

  private:
    // The mesh on which our data items are stored. This is a shared_ptr because
    // we retain long-term ownership but the localGrid() method lets other classes
    // obtain a pointer, and we don't know how long that reference will live. 
    std::shared_ptr<grid_type> _local_grid;

    // Data items returned from Cabana create methods. Even though we likely hold the
    // only pointers these objects, they are shared_ptr instead of uniq_ptr because 
    // Cabana returns shared_ptr.
    std::shared_ptr<cell_array> _liveness_curr, _liveness_next; // Data values
    std::shared_ptr<halo_type> _halo; // Communication pattern
};

} // namespace CabanaGOL

#endif // CABANAGOL_PROBLEMMANAGER_HPP
