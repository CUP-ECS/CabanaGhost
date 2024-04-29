/*
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 *
 * @section DESCRIPTION
 * 2 dimensional game of life with cabana-provided arrays, interation, and 
 * halo exchange primitives
 */

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements

#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <mpi.h>

// And now 
#include "Solver.hpp"

#if DEBUG
#include <iostream>
#endif

// Include Statements
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>

#include "ProblemManager.hpp"
using namespace Cabana::Grid;

// Short Args: n - Cell Count
// x - On-node Parallelism ( Serial/Threaded/OpenMP/CUDA ),
// t - Time Steps, F - Write Frequency
static char* shortargs = (char*)"n:t:x:F:h";

static option longargs[] = {
    // Basic simulation parameters
    { "ncells", required_argument, NULL, 'n' },
    { "timesteps", required_argument, NULL, 't' },
    { "driver", required_argument, NULL, 'x' },
    { "write-freq", required_argument, NULL, 'F' },
    { "help", no_argument, NULL, 'j' },
    { 0, 0, 0, 0 } };

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    std::string device; /**< ( Serial, Threads, OpenMP, CUDA ) */
    std::array<int, 2> global_num_cells;          /**< Number of cells */
    int t_final; /**< Ending time */
    int write_freq;     /**< Write frequency */
};

/**
 * Outputs help message explaining command line options.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
void help( const int rank, char* progname )
{
    if ( rank == 0 )
    {
        std::cout << "Usage: " << progname << "\n";
        std::cout << std::left << std::setw( 10 ) << "-x" << std::setw( 40 )
                  << "On-node Parallelism Model (default serial)" << std::left
                  << "\n";
        std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 )
                  << "Number of Cells (default 128)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
                  << "NUmber of timesteps to simulate (default 4.0)" 
                  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-F" << std::setw( 40 )
                  << "Write Frequency (default 20)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-h" << std::setw( 40 )
                  << "Print Help Message" << std::left << "\n";
    }
}

/**
 * Parses command line input and updates the command line variables
 * accordingly.
 * Usage: ./[program] [-h help] [-n number-of-cells]
 * [-t number-time-steps] [-F write-frequency] [-x thread-model]
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
int parseInput( const int rank, const int argc, char** argv, ClArgs& cl )
{

    /// Set default values

    cl.device = "serial"; // Default Thread Setting

    cl.t_final = 100;
    cl.write_freq = 1;
    cl.global_num_cells = { 128, 128 };

    int ch;
    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) !=
            -1 )
    {
        switch ( ch )
        {
        case 'n':
            cl.global_num_cells[0] = atoi( optarg );
            if ( cl.global_num_cells[0] <= 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid cell number argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            cl.global_num_cells[1] = cl.global_num_cells[0];
            break;
        case 't':
            cl.t_final = atoi( optarg );
            if ( cl.t_final <= 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid timesteps argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'x':
            cl.device = strdup( optarg );
            if ( ( cl.device.compare( "serial" ) != 0 ) &&
                 ( cl.device.compare( "cuda" ) != 0 ) &&
                 ( cl.device.compare( "openmp" ) != 0 ) &&
                 ( cl.device.compare( "pthreads" ) != 0 ) )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid  parallel device argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'F':
            cl.write_freq = atoi( optarg );
            if ( cl.write_freq < 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid write frequency argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'h':
            help( rank, argv[0] );
            exit( 0 );
            break;
        default:
            if ( rank == 0 )
            {
                std::cerr << "Invalid argument.\n";
                help( rank, argv[0] );
            }
            exit( -1 );
            break;
        }
    }

    // Return Successfully
    return 0;
}

// Initialize field to a constant quantity and velocity
template <class ArrayType>
struct MeshInitFunc
{
    using array_type = ArrayType;
    using view_type = typename array_type::view_type;

    view_type v;

    MeshInitFunc( )
    {
    };

    void setArray(std::shared_ptr<array_type> a) 
    {
        v = a->view();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int i, int j ) const
    {
        double liveness;
        /* We put a glider the in the middle of every 10 x 10 block. */
        switch ((i % 10) * 10 + j % 10) {
          case 33:
          case 34:
          case 44:
          case 45:
          case 53:
            liveness = 1.0; 
            break;
          default:
            liveness = 0.0;
            break;
        }
        v(i, j, 0) = liveness;
    };
};

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    // Parse Input
    ClArgs cl;
    if ( parseInput( rank, argc, argv, cl ) != 0 )
        return -1;

    // Only Rank 0 Prints Command Line Options
    if ( rank == 0 )
    {
        // Print Command Line Options
        std::cout << "Cabana Game of Life\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.device
                  << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Cells"
                  << ": " << std::setw( 8 ) << cl.global_num_cells[0]
                  << std::setw( 8 ) << cl.global_num_cells[1]
                  << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Total Simulation Time"
                  << ": " << std::setw( 8 ) << cl.t_final << "\n";
        std::cout << std::left << std::setw( 20 ) << "Write Frequency"
                  << ": " << std::setw( 8 ) << cl.write_freq
                  << "\n"; // Steps between write
        std::cout << "====================================\n";
    }

    // Call advection solver
    MeshInitFunc<Cabana::Grid::Array<double, Cabana::Grid::Cell, Cabana::Grid::UniformMesh<double, 2>>> initializer = {};
    auto solver = CabanaGhost::createSolver( cl.device, cl.global_num_cells, initializer );
    solver->solve(cl.t_final, cl.write_freq);

    // We need to make sure the solver, which includes a bunch of Kokkos-related allocations
    // is shut down before we shut down Kokkos. TO do that, we simply set teh solver pointer 
    // to nullptr; this will cause its reference count to go to zero and for it and everything
    // it points to to be freed.
    solver = nullptr;

    // Shut things down
    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
