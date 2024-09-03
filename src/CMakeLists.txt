set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo 
  MPI::MPI_CXX
  )

add_executable( gol GOL.cpp )
target_link_libraries( gol PRIVATE ${DEPENDS_ON} )
install(TARGETS gol DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable( jacobi Jacobi.cpp )
target_link_libraries( jacobi PRIVATE ${DEPENDS_ON} )
install(TARGETS jacobi DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(ENABLE_TESTS)
  blt_add_executable(NAME goltest
    SOURCES GOL.cpp
    DEPENDS_ON gtest Cabana::Core)
  blt_add_test(NAME goltest
    COMMAND goltest)

  blt_add_executable(NAME jacobitest
    SOURCES Jacobi.cpp
    DEPENDS_ON gtest Cabana::Core)
  blt_add_test(NAME jacobitest
    COMMAND jacobitest)
  
endif()
