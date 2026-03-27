set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo 
  MPI::MPI_CXX
  )

if( Cabana_ENABLE_STREAM-TRIGGERING )
  list(APPEND DEPENDS_ON
    MPIAdvance::stream-triggering
  )
endif()
  
add_executable( gol GOL.cpp )
target_link_libraries( gol PRIVATE ${DEPENDS_ON} )
install(TARGETS gol DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable( jacobi Jacobi.cpp )
target_link_libraries( jacobi PRIVATE ${DEPENDS_ON} )
install(TARGETS jacobi DESTINATION ${CMAKE_INSTALL_BINDIR})
