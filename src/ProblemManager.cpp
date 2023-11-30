/****************************************************************************
 * Copyright (c) 2023 by the CabanaGOL authors                              *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabanaGOL benchmark. CabanaGOL is               *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ProblemManager.hpp>

namespace CabanaGOL
{
//---------------------------------------------------------------------------//
template class ProblemManager<2, Kokkos::DefaultHostExecutionSpace,
                              Kokkos::HostSpace>;

//---------------------------------------------------------------------------//

} // end namespace CabanaGOL
