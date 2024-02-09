/**
 * @file Halo.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Extended Halo abstraction for Cabana that includes partitioning and specialiation
 * for a specific buffer.
 */

#ifndef CABANAGOL_HALO_HPP
#define CABANAGOL_HALO_HPP

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace CabanaGOL
{

template <class MemorySpace> class PersistentHalo;

// CabanaGOL Halos extend the original Cabana::Grid Halos with state for
// managing repeated partitioned exchanges on a specific Kokkos View. In 
// addition to the basic halos that Cabana supports, they can also serve 
// provide tiled, persistent halos objects with the specified pattern 
template <class MemorySpace>
class Halo : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    template <class Pattern, class... ArrayTypes>
    Halo(const Pattern & pattern, 
         const int width, const ArrayTypes&... arrays)
        : Cabana::Grid::Halo<MemorySpace>(pattern, width, arrays...)
    {
        // The buffers and steering are all created here, so we can go ahead and 
        // call MPI_PSend_Init
    }

    template <class ExecutionSpace, class... ArrayTypes>
    void gather(const ExecutionSpace &exec_space, const ArrayTypes&... arrays)
    {
        // Start the MPI_PSends/PRecvs
        // Run the appropriate packing kernel
        // PRready all the partitions as packing finishes
        // Wait for the PSends/PRecvs to finish
    }

    template <class ExecutionSpace, class... ArrayTypes>
    void gatherStart(const ExecutionSpace& exec_space, 
                     const ArrayTypes&... arrays)
    { 
        // Start the MPI_PSends
    }

    // General gatherReady that can be called outside of a parallel for
    template <class ExecutionSpace, std::size_t NumSpaceDims, class... ArrayTypes>
    void gatherReady(const ExecutionSpace& exec_space,
                     Kokkos::Array<int, NumSpaceDims> tile,
                     const ArrayTypes&... arrays)
    {
        // Run the appropriate packing kernel
        // PRready relevant partitions as packing finishes
        // XXX Ideally loop on parrived for the relevant buffers
        // XXX and run unpacking code! Moved to gatherFinish for now
    }

    // XXX We need a version of this specialized for thread teams XXX
    template <class ExecutionSpace, class MemberType,
              std::size_t NumSpaceDims, class... ArrayTypes>
    void gatherReady(const ExecutionSpace& exec_space,
                     const MemberType team_member,
                     Kokkos::Array<int, NumSpaceDims> tile,
                     const ArrayTypes&... arrays)
    {
        // Run the appropriate packing code using this thread team
        // PRready relevant partitions as packing finishes
        // XXX Ideally loop on parrived for the relevant buffers
        // XXX and run unpacking code! Moved to gatherFinish for now
    }

    template <class ExecutionSpace, class... ArrayTypes>
    void gatherFinish(const ExecutionSpace& exec_space, 
                      const ArrayTypes&... arrays)
    {
        // Run unpacking kernels for the received data (if we can't in Ready)
        // Wait for the PSends/PRecvs to finish
    }

  private:
    // State goes here, including buffers for packing partitions into 
    // and state for keeping track of which buffers have been sent and 
    // received. 
    enum {HALO_IDLE = 0, HALO_SCATTER, HALO_GATHER} halo_state;
};

/*!
  \brief Halo creation function.
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param arrays The arrays over which to build the halo.
  \return Shared pointer to a Halo.
*/
template <class Pattern, class... ArrayTypes>
auto createHalo( const Pattern& pattern, const int width,
                 const ArrayTypes&... arrays )
{
    using memory_space = typename Cabana::Grid::ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_unique<Halo<memory_space>>( pattern, width, arrays... );
}

} // CabanaGOL

#endif // CABANAGOL_HALO_HPP
