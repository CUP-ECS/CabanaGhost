# project settings
cmake_minimum_required(VERSION 3.18)

project(CabanaGhost LANGUAGES C CXX VERSION 0.1.0)

include(GNUInstallDirs)

# Set the C++ version we need - 20 for requires
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Get the dependent packages we need
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()
find_package(SILO REQUIRED)

# Get Clang Format to use for making sure the resulting code is
# properly formatted
find_package(CLANG_FORMAT 10)

# library
add_subdirectory(src)

# Add a target for formatting the code using Clang
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp tests/*.hpp tests/*.cpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

# Find MPI ADVANCE Stream Triggering
option(CabanaGhost_ENABLE_MPI_ADVANCE "Support MPI Advance backends, namely stream-triggering." OFF)
#Cabana_add_dependency( PACKAGE stream-triggering )
#set_package_properties(stream-triggering PROPERTIES TYPE OPTIONAL PURPOSE "Stream Triggered Backend")
if( CabanaGhost_ENABLE_MPI_ADVANCE )
  find_package(stream-triggering REQUIRED)
  if(stream-triggering_FOUND)# AND Cabana_ENABLE_MPISTREAM)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindStreamTriggering.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX} )
  endif()
endif()
