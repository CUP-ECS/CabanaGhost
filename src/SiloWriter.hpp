/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 *
 * @section DESCRIPTION
 * Silo Writer class to write results to a silo file using PMPIO
 */

#ifndef CABANAGHOST_SILOWRITER_HPP
#define CABANAGHOST_SILOWRITER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Grid.hpp>

#include "ProblemManager.hpp"

#include <pmpio.h>
#include <silo.h>

namespace CabanaGhost
{

/**
 * The SiloWriter Class
 * @class SiloWriter
 * @brief SiloWriter class to write results to Silo file using PMPIO
 **/
template <unsigned long Dims>
class SiloWriter
{
  public:
    using mem_type = typename Kokkos::DefaultExecutionSpace::memory_space;
    using mesh_type = Cabana::Grid::UniformMesh<double, Dims>;
    using grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using exec_type = Kokkos::DefaultExecutionSpace;
    using pm_type = ProblemManager<exec_type, Dims>;
    using device_type = typename exec_type::device_type;
    /**
     * Constructor
     * Create new SiloWriter
     *
     * @param pm Problem manager object
     */
    SiloWriter( const pm_type & pm )
        : _pm( pm )
    {
    };

    /* Helper types and functions for converting multiple dimensions of grids */
    using value_type = typename pm_type::cell_array_type::value_type;
    using view_data_type = std::conditional_t<
        3 == Dims, value_type****, std::conditional_t<2 == Dims, value_type***, void>>;
    using owned_view_type = Kokkos::View<view_data_type, Kokkos::LayoutLeft, 
        typename pm_type::cell_array_type::device_type>;
    owned_view_type allocateOwnedArray(Cabana::Grid::IndexSpace<Dims> d)
        requires (Dims == 3)
    {
        return owned_view_type("qOwned", d.extent( 0 ), d.extent( 1 ), d.extent( 2 ), 1);
    }
    owned_view_type allocateOwnedArray(Cabana::Grid::IndexSpace<Dims> d)
        requires (Dims == 2)
    {
        return owned_view_type("qOwned", d.extent( 0 ), d.extent( 1 ), 1);
    }

    struct CopyFunctor {
        typename pm_type::cell_array_type::view_type orig;
        owned_view_type owned; 
        int xmin, ymin, zmin;
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i, const int j, const int k) const
            requires (Dims == 3)
        {
                owned( i - xmin, j - ymin, k - zmin, 0 ) = orig( i, j, k, 0 );
        }
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i, const int j) const 
            requires (Dims == 2)
        {
            owned( i - xmin, j - ymin, 0 ) = orig( i, j, 0 );
        }
    };
    /**
     * Write File
     * @param dbile File handler to dbfile
     * @param name File name
     * @param time_step Current time step
     * @param time Current tim
     * @param dt Time Step (dt)
     * @brief Writes the locally-owned portion of the mesh/variables to a file
     **/
    void writeFile( DBfile* dbfile, char* meshname, int time_step, double time,
                    double dt )
    {
        // Initialize Variables
        int dims[Dims], zdims[Dims];
        double *coords[Dims]; 
        // double *spacing[Dims];
        const char* coordnames[3] = { "X", "Y", "Z" };
        DBoptlist* optlist;

        // Retrieve the Local Grid and Local Mesh
        std::shared_ptr<grid_type> local_grid = _pm.localGrid();
        Cabana::Grid::LocalMesh<mem_type, mesh_type> local_mesh = Cabana::Grid::createLocalMesh<mem_type>(*local_grid);

        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile" );

        // Set DB Options: Time Step, Time Stamp and Delta Time
        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::SetupOptions" );
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );
        int dbcoord = DB_CARTESIAN;
        DBAddOption( optlist, DBOPT_COORDSYS, &dbcoord );
        int dborder = DB_ROWMAJOR;
        DBAddOption( optlist, DBOPT_MAJORORDER, &dborder );
        Kokkos::Profiling::popRegion();

        // Get the size of the local cell space and declare the
        // coordinates of the portion of the mesh we're writing
        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::WriteMesh" );
        auto cell_domain = local_grid->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        for ( unsigned int i = 0; i < Dims; i++ )
        {
            zdims[i] = cell_domain.extent( i ); // zones (cells) in a dimension
            dims[i] = zdims[i] + 1;             // nodes in a dimension
        }

        // Allocate coordinate arrays in each dimension
        for ( unsigned int i = 0; i < Dims; i++ )
        {
            coords[i] = (double*)malloc( sizeof( double ) * dims[i] );
        }

        // Fill out coords[] arrays with coordinate values in each dimension
        for ( unsigned int d = 0; d < Dims; d++ )
        {
            for ( int i = cell_domain.min( d ); i < cell_domain.max( d ) + 1;
                  i++ )
            {
                int iown = i - cell_domain.min( d );
                int index[Dims];
                double location[Dims];
                for ( unsigned int j = 0; j < Dims; j++ )
                    index[j] = 0;
                index[d] = i;
                local_mesh.coordinates( Cabana::Grid::Node(), index, location );
                coords[d][iown] = location[d];
            }
        }

        DBPutQuadmesh( dbfile, meshname, (DBCAS_t)coordnames, coords, dims,
                       Dims, DB_DOUBLE, DB_COLLINEAR, optlist );
        Kokkos::Profiling::popRegion();

        // Now we write the individual variables associated with this
        // portion of the mesh, potentially copying them out of device space
        // and making sure not to write ghost values.

        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::WriteLiveness" );
        auto q =
            _pm.get( Cabana::Grid::Cell(), Field::Liveness(), Version::Current() ).view();
        auto xmin = cell_domain.min( 0 );
        auto ymin = cell_domain.min( 1 );
        auto zmin = Dims == 3 ? cell_domain.min( 2 ) : 0;

        // Silo is expecting row-major data so we make this a LayoutRight
        // array that we copy data into and then get a mirror view of.
        // XXX WHY DOES THIS ONLY WORK LAYOUTLEFT?
        owned_view_type qOwned = allocateOwnedArray(cell_domain);

        CopyFunctor copy_functor;
        copy_functor.orig = q; copy_functor.owned = qOwned;
        copy_functor.xmin = cell_domain.min(0); 
        copy_functor.ymin = cell_domain.min(1);
        copy_functor.zmin = Dims == 3 ? cell_domain.min(2) : 0;

        Kokkos::parallel_for( "SiloWriter::qowned copy",
            createExecutionPolicy( cell_domain, Kokkos::DefaultExecutionSpace() ),
            copy_functor);
        auto qHost =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qOwned );

        DBPutQuadvar1( dbfile, "liveness", meshname, qHost.data(), zdims, Dims,
                       NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );
        Kokkos::Profiling::popRegion();

        for ( unsigned int i = 0; i < Dims; i++ )
        {
            free( coords[i] );
        }

        // Free Option List
        DBFreeOptlist( optlist );
        Kokkos::Profiling::popRegion(); // writeFile region
    };

    /**
     * Create New Silo File for Current Time Step and Owning Group
     * @param filename Name of file
     * @param nsname Name of directory inside of the file
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void* createSiloFile( const char* filename, const char* nsname,
                                 void* user_data )
    {

        int driver = *( (int*)user_data );
        Kokkos::Profiling::pushRegion( "SiloWriter::CreateSiloFile" );

        DBfile* silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL,
                                      "CabanaGhostRaw", driver );

        if ( silo_file )
        {
            DBMkDir( silo_file, nsname );
            DBSetDir( silo_file, nsname );
        }

        Kokkos::Profiling::popRegion();

        return (void*)silo_file;
    };

    /**
     * Open Silo File
     * @param filename Name of file
     * @param nsname Name of directory inside of file
     * @param ioMode Read/Write/Append Mode
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void* openSiloFile( const char* filename, const char* nsname,
                               PMPIO_iomode_t ioMode,
                               [[maybe_unused]] void* user_data )
    {
        Kokkos::Profiling::pushRegion( "SiloWriter::openSiloFile" );
        DBfile* silo_file = DBOpen(
            filename, DB_UNKNOWN, ioMode == PMPIO_WRITE ? DB_APPEND : DB_READ );

        if ( silo_file )
        {
            if ( ioMode == PMPIO_WRITE )
            {
                DBMkDir( silo_file, nsname );
            }
            DBSetDir( silo_file, nsname );
        }
        Kokkos::Profiling::popRegion();
        return (void*)silo_file;
    };

    /**
     * Close Silo File
     * @param file File pointer
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void closeSiloFile( void* file, [[maybe_unused]] void* user_data )
    {
        Kokkos::Profiling::pushRegion( "SiloWriter::closeSiloFile" );
        DBfile* silo_file = (DBfile*)file;
        if ( silo_file )
            DBClose( silo_file );
        Kokkos::Profiling::popRegion();
    };

    /**
     * Write Multi Object Silo File the References Child Files in order to
     * have entire set of data for the time step writen by each rank in
     * a single logical file
     *
     * @param silo_file Pointer to the Silo File
     * @param baton Baton object from PMPIO
     * @param size Number of Ranks
     * @param time_step Current time step
     * @param file_ext File extension (PDB, HDF5)
     **/
    void writeMultiObjects( DBfile* silo_file, PMPIO_baton_t* baton, int size,
                            int time_step, const char* file_ext )
    {
        Kokkos::Profiling::pushRegion( "SiloWriter::writeMultiObjects" );
        char** mesh_block_names = (char**)malloc( size * sizeof( char* ) );
        char** q_block_names = (char**)malloc( size * sizeof( char* ) );

        int* block_types = (int*)malloc( size * sizeof( int ) );
        int* var_types = (int*)malloc( size * sizeof( int ) );

        DBSetDir( silo_file, "/" );

        for ( int i = 0; i < size; i++ )
        {
            int group_rank = PMPIO_GroupRank( baton, i );
            mesh_block_names[i] = (char*)malloc( 1024 );
            q_block_names[i] = (char*)malloc( 1024 );

            snprintf( mesh_block_names[i], 1024,
                     "raw/CabanaGhostOutput%05d%05d.%s:/domain_%05d/Mesh",
                     group_rank, time_step, file_ext, i );
            snprintf( q_block_names[i], 1024,
                     "raw/CabanaGhostOutput%05d%05d.%s:/domain_%05d/liveness",
                     group_rank, time_step, file_ext, i );
            block_types[i] = DB_QUADMESH;
            var_types[i] = DB_QUADVAR;
        }

        DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names,
                        block_types, 0 );
        DBPutMultivar( silo_file, "multi_liveness", size, q_block_names,
                       var_types, 0 );
        for ( int i = 0; i < size; i++ )
        {
            free( mesh_block_names[i] );
            free( q_block_names[i] );
        }

        free( mesh_block_names );
        free( q_block_names );
        free( block_types );
        free( var_types );
        Kokkos::Profiling::popRegion();
    }

    // Function to Create New DB File for Current Time Step
    /**
     * Createe New DB File for Current Time Step
     * @param name Name of directory in silo file
     * @param time_step Current time step
     * @param time Current time
     * @param dt Time step (dt)
     **/
    void siloWrite( char* name, int time_step, double time, double dt )
    {
        // Initalize Variables
        DBfile* silo_file;
        DBfile* master_file;
        int size, rank;
        MPI_Comm comm;
        int driver = DB_PDB;
        const char* file_ext = "silo";
        // TODO: Make the Number of Groups a Constant or a Runtime Parameter (
        // Between 8 and 64 )
        int numGroups = 2;
        char masterfilename[256], filename[256], nsname[256];
        PMPIO_baton_t* baton;

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite" );

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::Setup" );
        comm = _pm.localGrid()->globalGrid().comm();
        MPI_Comm_size( comm, &size );
        MPI_Comm_rank( comm, &rank );
        MPI_Bcast( &numGroups, 1, MPI_INT, 0, comm );
        MPI_Bcast( &driver, 1, MPI_INT, 0, comm );

        baton =
            PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1,
                        createSiloFile, openSiloFile, closeSiloFile, &driver );

        // Set Filename to Reflect TimeStep
        snprintf( masterfilename, 256, "data/CabanaGhost%05d.%s", time_step,
                 file_ext );
        snprintf( filename, 256, "data/raw/CabanaGhostOutput%05d%05d.%s",
                 PMPIO_GroupRank( baton, rank ), time_step,
                 file_ext );
        snprintf( nsname, 256, "domain_%05d", rank );

        // Show Errors and Force FLoating Point
        DBShowErrors( DB_ALL, NULL );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::batonWait" );
        silo_file = (DBfile*)PMPIO_WaitForBaton( baton, filename, nsname );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::writeState" );
        writeFile( silo_file, name, time_step, time, dt );
        if ( rank == 0 )
        {
            master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL,
                                    "CabanaGhost", driver );
            writeMultiObjects( master_file, baton, size, time_step, "silo" );
            DBClose( master_file );
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::batonHandoff" );
        PMPIO_HandOffBaton( baton, silo_file );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::finish" );
        PMPIO_Finish( baton );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::popRegion(); // siloWrite
    }

  private:
    // The problem manager is owned by
    const pm_type & _pm;
};

}; // namespace CabanaGhost
#endif
