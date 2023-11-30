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

#include <Mesh.hpp>

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
template <std::size_t NumSpaceDim, class ExecutionSpace, class MemorySpace>
class ProblemManager;

/* The 2D implementation of hte problem manager class */
template <class ExecutionSpace, class MemorySpace>
class ProblemManager<2, ExecutionSpace, MemorySpace>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using Cell = Cabana::Grid::Cell;
    using FaceI = Cabana::Grid::Face<Cabana::Grid::Dim::I>;
    using FaceJ = Cabana::Grid::Face<Cabana::Grid::Dim::J>;
    using FaceK = Cabana::Grid::Face<Cabana::Grid::Dim::K>;

    using cell_array =
        Cabana::Grid::Array<double, Cabana::Grid::Cell, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
    using halo_type = Cabana::Grid::Halo<MemorySpace>;
    using mesh_type = Mesh<2, ExecutionSpace, MemorySpace>;

    template <class InitFunc>
    ProblemManager( const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associuated with them?
        auto iface_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Face<Cabana::Grid::Dim::I>() );
        auto jface_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Face<Cabana::Grid::Dim::J>() );
        auto cell_scalar_layout =
            Cabana::Grid::createArrayLayout( _mesh->localGrid(), 1, Cabana::Grid::Cell() );

        // The actual arrays storing mesh quantities
        _liveness_curr = Cabana::Grid::createArray<double, MemorySpace>(
            "liveness", cell_scalar_layout );
        _liveness_next = Cabana::Grid::createArray<double, MemorySpace>(
            "liveness", cell_scalar_layout );
        Cabana::Grid::ArrayOp::assign( *_liveness_curr, 0.0, Cabana::Grid::Ghost() );
        Cabana::Grid::ArrayOp::assign( *_liveness_next, 0.0, Cabana::Grid::Ghost() );

        // Halo patterns for the just liveness. This halo is just one cell deep,
        // as we only look at that much data to calculate changes in state.
        int halo_depth = _mesh->localGrid()->haloCellWidth();
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
        // DEBUG: Trace State Initialization
        if ( _mesh->rank() == 0 && DEBUG )
            std::cout << "Initializing Cell Fields\n";

        // Get Local Grid and Local Mesh
        auto local_grid = *( _mesh->localGrid() );
        auto local_mesh = *( _mesh->localMesh() );
        double cell_size = _mesh->cellSize();

        // Get State Arrays
        auto l = get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );

        // Loop Over All Owned Cells ( i, j )
        auto own_cells = local_grid.indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(),
                                                Cabana::Grid::Local() );
        int index[2] = { 0, 0 };
        double loc[2]; // x/y loocation of the cell at 0, 0
        local_mesh.coordinates( Cabana::Grid::Cell(), index, loc );
        Kokkos::parallel_for(
            "Initialize Cells`",
            Cabana::Grid::createExecutionPolicy( own_cells, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Get Coordinates Associated with Indices ( i, j )
                int coords[2] = { i, j };
                double x[2];
                x[0] = loc[0] + cell_size * i;
                x[1] = loc[1] + cell_size * j;
                // Initialization Function
                create_functor( Cabana::Grid::Cell(), Field::Liveness(), coords, x,
                                l( i, j, 0 ) );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>>& mesh() const
    {
        return _mesh;
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
        _advection_halo->gather( ExecutionSpace(), *_liveness_curr )
    };
    void gather( Version::Next ) const
    {
        _halo->gather( ExecutionSpace(), *_liveness_next );
    };

  private:
    // The mesh on which our data items are stored
    std::shared_ptr<mesh_type> _mesh;

    // Basic long-term quantities stored in the mesh
    std::shared_ptr<cell_array> _liveness_curr, _liveness_next;

    // Halo communication pattern for the mesh quantities
    std::shared_ptr<halo_type> _halo;
};

} // namespace CabanaGOL

#endif // CABANAGOL_PROBLEMMANAGER_HPP
