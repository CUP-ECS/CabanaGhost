/****************************************************************************
 * Copyright (c) 2021 by the CabanaGOL authors                              *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabanaGOL benchmark. CabanaGOL is               *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Mesh.hpp>

namespace CabanaGOL
{
//---------------------------------------------------------------------------//
#ifdef KOKKOS_ENABLE_CUDA
template class Mesh<2, Kokkos::Cuda, Kokkos::CudaSpace>;
#else
template class Mesh<2, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace CabanaGOL
