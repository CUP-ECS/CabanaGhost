/*
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Evan Drake Suggs <esuggs@tntech.edu>
 *
 * @section DESCRIPTION
 * 2 dimensional game of life with cabana-provided arrays, interation, and 
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
    std::array<int, 2> global_num_cells;          /**< Number of cells */
    int t_final; /**< Ending time */
    int write_freq;     /**< Write frequency */
};

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    MeshInitFunc( )
    {
    };

    KOKKOS_INLINE_FUNCTION
    double operator()( int index[2], double coords[2] ) const
    {
        int i = coords[0], j = coords[1];
        double liveness;
        /* We put a glider the in the middle of every 10 x 10 block. */
        switch ((i % 10) * 10 + j % 10) {
          case 33:
          case 34:
          case 44:
          case 45:
          case 53:
            return 1.0; 
            break;
          default:
            return 0.0;
            break;
        }
        return 0.0;
    };
};

struct GOL2DFunctor {
    using view_type = Kokkos::View<double ***>;
    view_type _src_array, _dst_array;

    void setViews(const view_type s, const view_type d) {
        _src_array = s;
            _dst_array = d;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
        double sum = 0.0;
        int ii, jj;
        for (ii = -1; ii <= 1; ii++) {
            for (jj = -1; jj <= 1; jj++) {
                if ((ii == 0) && (jj == 0)) continue;
                sum += _src_array(i + ii,j + jj, 0);
            }
        }
        if (_src_array(i, j, 0) == 0.0) {
            if ((sum > 2.99) && (sum < 3.01))
                _dst_array(i, j, 0) = 1.0;
            else 
                _dst_array(i, j, 0) = 0.0;
        } else {
            if ((sum >= 1.99) && (sum <= 3.01))
                _dst_array(i, j, 0) = 1.0;
            else 
                _dst_array(i, j, 0) = 0.0;
        }
    };
    GOL2DFunctor() {}
};

TEST(goltest, BasicTest){
  int comm_size, rank;
  int test = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank
  ASSERT_GE(comm_size, rank);
  ClArgs cl;
  cl.t_final = 100;
  cl.write_freq = 0;
  cl.global_num_cells = { 128, 128 };
  ASSERT_EQ(cl.t_final, 100);
  Kokkos::Timer timer;
  {
    using namespace CabanaGhost;
    MeshInitFunc initializer;
    GOL2DFunctor gol2Dfunctor;
    Solver<2, GOL2DFunctor, Approach::Flat, Approach::Host> 
      solver( cl.global_num_cells, true, gol2Dfunctor, initializer );
    solver.solve(cl.t_final, 0.0, cl.write_freq); 
  }
  double time = timer.seconds();
  std::cout << "Time: " << time << std::endl;
}
