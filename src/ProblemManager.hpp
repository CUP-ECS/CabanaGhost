/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Problem manager class that stores the mesh and the state data and performs
 * scatters and gathers
 */

#ifndef CABANAGHOST_PROBLEMMANAGER_HPP
#define CABANAGHOST_PROBLEMMANAGER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace CabanaGhost
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
template <class ExecutionSpace, unsigned long Dims>
class ProblemManager
{
  public:
    using global_mesh_type = Cabana::Grid::UniformMesh<double, Dims>;
    using cell_array_type = Cabana::Grid::Array<double, Cabana::Grid::Cell, global_mesh_type>;
    using memory_space = typename cell_array_type::memory_space;
    using view_type = typename cell_array_type::view_type;
    using grid_type = Cabana::Grid::LocalGrid<global_mesh_type>;
    using local_mesh_type = Cabana::Grid::LocalMesh<memory_space, global_mesh_type>;
    using exec_space = ExecutionSpace;
    using halo_type = Cabana::Grid::Experimental::StreamHalo<exec_space, memory_space>;

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
        _liveness_curr = Cabana::Grid::createArray<double>(
            "liveness", cell_scalar_layout );
        _liveness_next = Cabana::Grid::createArray<double>(
            "liveness", cell_scalar_layout );
        //Cabana::Grid::ArrayOp::assign( *_liveness_curr, 0.0, Cabana::Grid::Ghost() );
        //Cabana::Grid::ArrayOp::assign( *_liveness_next, 0.0, Cabana::Grid::Ghost() );

        // Halo patterns for the just liveness. This halo is just one cell deep,
        // as we only look at that much data to calculate changes in state.
        // First we create the generic halo pattern itself which can
        // handle non-persistent halos 
        int halo_depth = _local_grid->haloCellWidth();
        _halo = Cabana::Grid::Experimental::createStreamHalo( exec_space(), 
            Cabana::Grid::NodeHaloPattern<Dims>(), halo_depth, 
            *_liveness_curr );

        // Initialize State Values ( liveness )
        initialize( create_functor );
    }

    template <class ViewType, class CellFunctor, class LocalMesh>
    struct ViewFunctor {
        CellFunctor _f;
        ViewType _v;
        LocalMesh _m;
        ViewFunctor(ViewType v, CellFunctor f, LocalMesh m) 
            : _v(v), _f(f), _m(m) {
        };
        KOKKOS_INLINE_FUNCTION
        void operator()( const int i, const int j ) const
            requires (Dims == 2)
        {
            int index[Dims] = {i, j};
            double coords[Dims];
	    _m.coordinates( Cabana::Grid::Cell(), index, coords);
            _v(i, j, 0) = _f(index, coords);
        };
        KOKKOS_INLINE_FUNCTION
        void operator()( const int i, const int j, const int k ) const
            requires (Dims == 3)
        {
            int index[Dims] = {i, j, k};
            double coords[Dims];
	    _m.coordinates( Cabana::Grid::Cell(), index, coords);
            _v(i, j, k, 0) = _f(index, coords);
        };
    };

    /**
     * Initializes state values in the cells by calling functor with local index and 
     * global coordinate
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initializeOwned( cell_array_type a, const InitFunctor& create_functor )
    {
        local_mesh_type local_mesh(*_local_grid);
        view_type v = a.view();
        ViewFunctor<view_type, InitFunctor, local_mesh_type> vf(v, create_functor, local_mesh);

        // Loop Over and initialize all owned cells ( i, j )
        auto own_cells = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(),
                                                  Cabana::Grid::Local() );

        Cabana::Grid::grid_parallel_for( "Initialize Boundaries", 
            exec_space(), own_cells, vf );
    }

    template <class InitFunctor>
    void initializeBoundary( cell_array_type a,  const InitFunctor& create_functor )
    {
        local_mesh_type local_mesh(*_local_grid);
        view_type v = a.view();
        ViewFunctor<view_type, InitFunctor, local_mesh_type> vf(v, create_functor, local_mesh);

        // We also initialize any boundary (non-periodic) ghost cells.
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    std::array<int, Dims> dir;
                    dir[0] = i; dir[1] = j; 
                    if (Dims == 3) {
                        if (i == j && j == k && k == 0) continue;
                        dir[2] = k; // Set the k direction if there is one
                    } else {
                        if (i == j && i == 0) continue; // skip no direction case
                        if (k != 0) continue; // skip redundant k cases
                    }
                    auto boundary_cells = 
                        _local_grid->boundaryIndexSpace( Cabana::Grid::Ghost(), 
                                                         Cabana::Grid::Cell(), dir);
                    Cabana::Grid::grid_parallel_for( "Initialize Boundaries", 
                        exec_space(), boundary_cells, vf );
                 }
            }
        }
    }

    template <class InitFunctor>
    void initialize( const InitFunctor& create_functor )
    {

        // Get State Arrays
        cell_array_type curr = get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() );
        cell_array_type next = get( Cabana::Grid::Cell(), Field::Liveness(), Version::Next() );

	initializeOwned(curr, create_functor);
        initializeBoundary(curr, create_functor);
        initializeBoundary(next, create_functor);
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
     * @return Returns array of current liveness at cell centers
     **/
    cell_array_type get( Cabana::Grid::Cell, Field::Liveness,
                    Version::Current ) const
    {
        return *_liveness_curr;
    };

    /**
     * Return Liveness Field
     * @param Location::Cell
     * @param Field::Liveness
     * @param Version::Next
     * @return Returns array of next liveness at cell centers
     **/
    cell_array_type get( Cabana::Grid::Cell, Field::Liveness,
                    Version::Next ) const
    {
        return *_liveness_next;
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
     * Gather State Data from Neighbors
     * @param Version
     **/
    void gather( Version::Current ) const
    {
        _halo->gather( exec_space(), *_liveness_curr );
    };
    void gather( Version::Next ) const
    {
        _halo->gather( exec_space(), *_liveness_next );
    };

    /** 
     * Stream-triggered gather from neighbors
     */
    void enqueueGather( Version::Current ) const
    {
        _halo->enqueueGather( *_liveness_curr );
    };
    void enqueueGather( Version::Next ) const
    {
        _halo->enqueueGather( *_liveness_next );
    };

    /**
     * Provide persistent halo objects (by value!) for making fine-grain 
     * exchanges
     * @param Version
     **/
    halo_type halo() const
    {
       return *_halo;
    };

  private:
    // The mesh on which our data items are stored. This is a shared_ptr because
    // we retain long-term ownership but the localGrid() method lets other classes
    // obtain a pointer, and we don't know how long that reference will live. 
    std::shared_ptr<grid_type> _local_grid;

    // Data items returned from Cabana create methods. Even though we likely hold the
    // only pointers these objects, they are shared_ptr instead of uniq_ptr because 
    // Cabana returns shared_ptr.
    std::shared_ptr<cell_array_type> _liveness_curr, _liveness_next; // Data values
    std::shared_ptr<halo_type> _halo; // Persistent halos
};

} // namespace CabanaGhost

#endif // CABANAGHOST_PROBLEMMANAGER_HPP
