/**
 * @file PartitionedHalo.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Extended Halo abstraction for Cabana that includes partitioning 
 */

#ifndef CABANAGOL_HALO_HPP
#define CABANAGOL_HALO_HPP

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

namespace CabanaGOL
{

// CabanaGOL Halos extend the original Cabana::Grid Halos with state for
// managing repeated partitioned exchanges on a specific Kokkos View. In 
// addition to the basic halos that Cabana supports, they can also serve 
// provide tiled, persistent halos objects with the specified pattern 
template <class MemorySpace>
class PartitionedHalo : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    using CabanaHalo = Cabana::Grid::Halo<MemorySpace>;
    // COmpute the information the number and size of each partition
    // so that we can create the appropriate partitioned sends
    void buildPartitionedCommData() 
    {
        // XXX For now, we use one partition per buffer and 
        // pReady in gatherFinish. Add proper partitioning later XXX
    }

    // Create MPI Handles 
    void buildMPIHandles() 
    {
        const int mpi_tag = 1234;
        MPI_Request r;
        for (int i = 0; i < CabanaHalo::_neighbor_ranks.size(); i++) {
            MPI_Precv_init(CabanaHalo::_ghosted_buffers[i].data(), 1, 
                           CabanaHalo::_ghosted_buffers[i].size(), MPI_BYTE, 
                           CabanaHalo::_neighbor_ranks[i], 
                           mpi_tag + CabanaHalo::_receive_tags[i], 
                           _comm, MPI_INFO_NULL, &r);
            _comm_handles.push_back(r);
        }
        for (int i = 0; i < CabanaHalo::_neighbor_ranks.size(); i++) {
            MPI_Psend_init(CabanaHalo::_owned_buffers[i].data(), 1, 
                           CabanaHalo::_owned_buffers[i].size(), MPI_BYTE, 
                           CabanaHalo::_neighbor_ranks[i], 
                           mpi_tag + CabanaHalo::_send_tags[i], 
                           _comm, MPI_INFO_NULL, &r);
            _comm_handles.push_back(r);
        }
    }

    template <class Pattern, class ArrayTypes>
    PartitionedHalo(const Pattern & pattern, const int width, 
//                    Kokkos::Array<int, NumSpaceDims> tiling,
                    const ArrayTypes& array)
        : Cabana::Grid::Halo<MemorySpace>(pattern, width, array)
    {
        _comm = CabanaHalo::getComm(array);

        // XXX need a logic assert that the pattern dimension and the NumSpaceDims 
        // are the same once we add that XXX
        // const std::size_t num_space_dim = Pattern::num_space_dim;

        // The buffers and steering for the exchange are all created 
        // in the superclass. We use those, and code above adds 
        // interfaces to access the Cabana private variables we need

        // Compute the sizes and limits of the tiling in each dimension
        // for each array layouts/space
        buildPartitionedCommData();
        
        // Call PSendInit and PRecvInit for the owned and ghost buffers
        buildMPIHandles();
    }

    ~PartitionedHalo() {
        // Free the initialized PSends and PRecvs. The superclass destructor
        // will then deallocate the buffers they use. XXX What to do if a 
        // partitioned gather is still active?
        // XXX releaseMPIHandles(); XXX
    }

    template <class ExecutionSpace, class ArrayTypes>
    void gatherStart(const ExecutionSpace& exec_space, 
                     const ArrayTypes& array)
    { 
        MPI_Request r;
        /* Start all the sends and receives - switch to a better C++ loop construct XXX */
        for (int i = 0; i < _comm_handles.size(); i++) {
            MPI_Start(&_comm_handles[i]);
        }
    }

    // General gatherReady that can be called outside of a parallel for. 
    // XXX How do we assert that we're invoked from host space?
    template <class ExecutionSpace, std::size_t NumSpaceDims, class ArrayTypes>
    void gatherReady(const ExecutionSpace& exec_space,
                     Kokkos::Array<int, NumSpaceDims> tile,
                     const ArrayTypes& array)
    {
        // Launch the appropriate packing kernel for this tile. Because this 
        // launces a kernel *and* syncs on it, we're going to be slow.

        // Pready relevant partition of owned buffer as packing finishes

        // XXX Ideally loop on parrived for the relevant buffers
        // XXX and run unpacking code! Moved to gatherFinish for now
    }

    // XXX We need a version of this specialized for thread teams XXX
    template <class ExecutionSpace, class MemberType,
              std::size_t NumSpaceDims, class ArrayTypes>
    void gatherReady(const ExecutionSpace& exec_space,
                     const MemberType team_member,
                     Kokkos::Array<int, NumSpaceDims> tile,
                     const ArrayTypes& array)
    {
        // 1. For the array we're packing/readying, work our way through
        // the steering buffer 
 
        //     1a. Get the index space we need and figure out the part of itthe portion of the buffers we need to pack based on the tile
        // that finished

        // 2. Figure out the range of _owned_steering buffer 
        // Run the appropriate packing code using this thread team
        // PRready relevant partitions as packing finishes
        // XXX Ideally loop on parrived for the relevant buffers
        // XXX and run unpacking code! Moved to gatherFinish for now
    }

    template <class ExecutionSpace, class ArrayTypes>
    void gatherFinish(const ExecutionSpace& exec_space, 
                      const ArrayTypes& array)
    {
        // XXX convert loop to C++ iterator format
        int s = _comm_handles.size();

        // Sends are the second half of comm handles.
        for (int i = s/2; i < s; i++) {
            // Since we only have one partition per buffer for now
            MPI_Pready(_comm_handles[i], 0);
        }

        // XXX Run unpacking kernels for the received data (if we can't in Ready) XXX

        // Wait for the PSends/PRecvs to finish
        MPI_Waitall(_comm_handles.size(), _comm_handles.data(), MPI_STATUSES_IGNORE);
    }

    template <class ExecutionSpace, class ArrayTypes>
    void gather(const ExecutionSpace &exec_space, const ArrayTypes& array)
    {
        gatherStart(exec_space, array); 
        // It would likely be faster to just launch a single packing kernel
        // and do bulk sends, but we use gatherReady() above as a first debug.

        // Finish the
        gatherFinish(exec_space, array);

        // Start a parallel for loop for each partition */
        // Run the appropriate packing kernel
        // PRready all the partitions as packing finishes
        // Wait for the PSends/PRecvs to finish
    }


  private:
    // State goes here, including buffers for packing partitions into 
    // and state for keeping track of which buffers have been sent and 
    // received. 
    enum {HALO_IDLE = 0, HALO_SCATTER, HALO_GATHER} halo_state;
    std::vector<MPI_Request> _comm_handles;
    MPI_Comm _comm; // XXX Saved to void having to pull it from the array each time
};

/*!
  \brief Halo creation function.
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param array The arrays over which to build the halo - only one for now.
  \return Shared pointer to a Halo.
*/
template <class Pattern, class ArrayTypes>
auto createPartitionedHalo( const Pattern& pattern, const int width,
                            const ArrayTypes& array )
{
    using memory_space = typename ArrayTypes::memory_space;
    return std::make_unique<PartitionedHalo<memory_space>>( pattern, width, array );
}

} // CabanaGOL

#endif // CABANAGOL_HALO_HPP
