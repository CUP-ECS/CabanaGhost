/****************************************************************************
 * Copyright (c) 2018-2023 by the CabanaGhost authors                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaGhost. CabanaGhost is distributed under       *
 * a BSD 3-clause license. For the licensing terms see the LICENSE file in  *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file NestedParallel.hpp
  \brief Abstractions for nested parallelism
*/
#ifndef CABANAGHOST_NESTEDPARALLEL_HPP
#define CABANAGHOST_NESTEDPARALLEL_HPP

#include <Kokkos_Core.hpp>

namespace CabanaGhost
{
//---------------------------------------------------------------------------//
/*!
  \brief Multi-dimensional Team Policy for Kokkos
 */
template <long N, Kokkos::Iterate OuterDir, class ExecutionSpace, class... Properties> 
class MDTeamPolicy : public TeamPolicy<ExecutionSpace, Properties...>
{
  
  public:
    MDTeamPolicy(ExecutionSpace & space, std::array<N, int> league_size, int team_size, int vector_length = 1) 
        : Kokkos::TeamPolicy(space, league_size.
}

/*!
  \brief Nested index space.
 */
template <class ExecutionSpace, long N, class OuterIterationPolicy, class InnerIterationPolicy>
class NestedIndexSpace : public Cabana::Grid::IndexSpace<N>
{
  public:
    using member_type = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
    template <typename Scalar>
    NestedIndexSpace( const Cabana::Grid::IndexSpace<N> &is, std::array<long, N>blocks )
        : Cabana::Grid::IndexSpace<N>(is)
    {
        std::copy(blocks.begin(), blocks.end(), _blocks.data() );
    }
 
    int leagueSize()
    {
        int l = 1;
        for (int i = 0; i < N; i++)
        {
            l *= _blocks[i];
        }
        return l;
    }

    //int itile = league_rank / blocks_per_dim,
    //    jtile = league_rank % blocks_per_dim;
    //int ibase = istart + itile * block_size,
    //    jbase = jstart + jtile * block_size;
    //int ilimit = std::min(ibase + block_size, iend),
    //    jlimit = std::min(jbase + block_size, jend);
    //int iextent = ilimit - ibase,
    //    jextent = jlimit - jbase;
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, N> tileNumber(member_type team_member)
    {
        /* Convert league rank into tile indexes 
         * XXX At some point, this should correspond to the view layout
         * and we can hopefully use ViewLayout/ViewMapping classes to
         * do what we need to do. */
        Kokkos::Array<long, N> zeroarray(0);
        return zeroarray;
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, N> tileStart(member_type team_member)
    {
        Kokkos::Array<long, N> zeroarray(0);
        return zeroarray;
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, N> tileExtent(member_type team_member)
    {
        Kokkos::Array<long, N> zeroarray(0);
        return zeroarray;

//        Kokkos::Array<long, N> extent;
//        for (int i = 0; i < N; i++)
//            extent[i] = _max[i] - _min[i];
//        return extent;
    }

  protected:
    Kokkos::Array<long, N> _blocks;
};


/*! 
 * createExecutionPolicy on a nested index space
 */
template <class ExecutionSpace, int N>
Kokkos::TeamPolicy<ExecutionSpace>
createNestedExecutionPolicy( const NestedIndexSpace<ExecutionSpace, N>& index_space )
{
    return Kokkos::TeamPolicy<ExecutionSpace>(index_space.leagueSize(), Kokkos::AUTO);
}

/*! 
 * nested_parallel_for a nested index space
 */
template <class MemberType, class FunctorType, class ExecutionSpace, int N>
void 
nested_parallel_for( const std::string &label, NestedIndexSpace<ExecutionSpace, N> index_space,
                     MemberType team_member, FunctorType &functor)
{
{
    return Kokkos::TeamPolicy<ExecutionSpace>(index_space.leagueSize(), Kokkos::AUTO);
}

/*! 
 * nested_parallel_for a nested index space
 */
template <class MemberType, class FunctorType, class ExecutionSpace, int N>
void 
nested_parallel_for( const std::string &label, NestedIndexSpace<ExecutionSpace, N> index_space,
                     MemberType team_member, FunctorType &functor)
{

} 

} // namespace CabanaGhost

#endif CABANAGHOST_NESTEDPARALLEL_HPP
