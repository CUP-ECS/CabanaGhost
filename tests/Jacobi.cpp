/*
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Evan Drake Suggs <esuggs@tntech.edu>
 *
 * @section DESCRIPTION
 * 3 dimensional jacobi iteration with cabana-provided arrays, interation, and 
 * halo exchange primitives
 */

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <mpi.h>

// And now 
#include "Solver.hpp"
#include "tstDriver.hpp"

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

TEST(goltest, BasicTest){
  int comm_size, rank;
  int test = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank
  ASSERT_GE(comm_size, rank);
  ClArgs cl;
  cl.max_iterations = 1000;
  cl.write_freq = 0;
  cl.global_num_cells = { 64, 64, 64 };
  cl.tolerance = 0.001;
  ASSERT_EQ(cl.max_iterations, 1000);
  //  ASSERT_EQ(c.global_num_cells[0], 128);
  Kokkos::Timer timer;
  {
    using namespace CabanaGhost;
    // Call advection solver
    MeshInitFunc initializer;
    JacobiFunctor iteration_functor;
    Solver<3, JacobiFunctor, Approach::Flat, Approach::Host> 
      solver(cl.global_num_cells, false, iteration_functor, initializer );
    solver.solve(cl.max_iterations, cl.tolerance, cl.write_freq);
  }
  double time = timer.seconds();
  std::cout << "Time: " << time << std::endl;
}
