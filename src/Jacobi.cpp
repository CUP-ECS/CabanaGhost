/*
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 *
 * @section DESCRIPTION
 * 3 dimensional jacobi iteration with cabana-provided arrays, interation, and 
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
static char* shortargs = (char*)"n:m:t:F:h";

static option longargs[] = {
    // Basic simulation parameters
    { "num-cells", required_argument, NULL, 'n' },
    { "max-iterations", required_argument, NULL, 'm' },
    { "tolerance", required_argument, NULL, 't' },
    { "write-freq", required_argument, NULL, 'F' },
    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    std::string device; /**< ( Serial, Threads, OpenMP, CUDA ) */
    std::array<int, 3> global_num_cells;          /**< Number of cells */
    int max_iterations; /**< Ending time */
    double tolerance;       /**< Convergence criteria */
    int write_freq;      /**< Write frequency */
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
        std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 )
                  << "Number of Cells (default 128)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-m" << std::setw( 40 )
                  << "Max number of iterations to calculate (default 1000)" 
                  << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
                  << "Convergence tolerance (default 0.001)" 
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
 * Usage: ./[program] [-h help] [-n number-of-cells] [-t max-time-steps] 
 *                    [-T tolerance] [-F write-frequency] 
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
int parseInput( const int rank, const int argc, char** argv, ClArgs& cl )
{

    /// Set default values
    cl.max_iterations = 1000;
    cl.write_freq = 0;
    cl.global_num_cells = { 64, 64, 64 };
    cl.tolerance = 0.001;

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
            cl.global_num_cells[2] = cl.global_num_cells[0];
            break;
        case 'm':
            cl.max_iterations = atoi( optarg );
            if ( cl.max_iterations <= 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid timesteps argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 't':
            cl.tolerance = atof( optarg );
            if ( cl.tolerance <= 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid timesteps argument.\n";
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
struct MeshInitFunc
{
    MeshInitFunc( )
    {
    };

    KOKKOS_INLINE_FUNCTION
    double operator()( const int index[3], const double coords[3] ) const
    {
        int d;

        //std::cout << "Initializing index "
        //          << "(" << index[0] << ", " << index[1] << ", " << index[2] << ")"
        //          << " with coordinate " 
        //          << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")"
        //          << "\n";
        for (d = 0; d < 3; d++)
            if (coords[d] < 0.0) return 100.0;

        return 0.0;
    };
};

struct JacobiFunctor {
    using view_type = Kokkos::View<double ****>;
    view_type _src_array, _dst_array;

    void setViews(const view_type s, const view_type d) {
        _src_array = s;
            _dst_array = d;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
        double sum = 0.0;
        int ii, jj, kk;
        for (ii = -1; ii <= 1; ii++) {
            for (jj = -1; jj <= 1; jj++) {
                for (kk = -1; kk <= 1; kk++) {
		    if ((ii == jj) && (jj == kk)) continue;
                    sum += _src_array(i + ii, j + jj, k + kk, 0);
                }
            }
        }
        _dst_array(i, j, k, 0) = sum / 26.0;
    };
    JacobiFunctor() {}
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
        std::cout << "Cabana Jacobi Iteration\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.device
                  << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Cells"
                  << ": " << std::setw( 8 ) << cl.global_num_cells[0]
                  << std::setw( 8 ) << cl.global_num_cells[1]
                  << std::setw( 8 ) << cl.global_num_cells[2]
                  << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Tolerance"
                  << ": " << std::setw( 8 ) << cl.tolerance << "\n";
        std::cout << std::left << std::setw( 20 ) << "Max Iterations"
                  << ": " << std::setw( 8 ) << cl.max_iterations << "\n";
        std::cout << std::left << std::setw( 20 ) << "Write Frequency"
                  << ": " << std::setw( 8 ) << cl.write_freq
                  << "\n"; // Steps between write
        std::cout << "====================================\n";
    }
    Kokkos::Timer timer;
    {
        using namespace CabanaGhost;
        // Call advection solver
        MeshInitFunc initializer;
        JacobiFunctor iteration_functor;
	// set Approach to Stream if using stream-triggering
        Solver<Kokkos::DefaultExecutionSpace, 3, JacobiFunctor, 
               Approach::Flat, Approach::Stream> 
            solver(cl.global_num_cells, false, iteration_functor, initializer );
        solver.solve(cl.max_iterations, cl.tolerance, cl.write_freq);
    }
    if ( rank == 0 )
      {
	double time = timer.seconds();
	std::cout << "Solver time: " << time << std::endl;
      }
    // Shut things down
    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
