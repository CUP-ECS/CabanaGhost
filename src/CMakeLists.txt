set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo 
  MPI::MPI_C
  )

add_executable( gol GOL.cpp )
target_link_libraries( gol PRIVATE ${DEPENDS_ON} )
install(TARGETS gol DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable( jacobi Jacobi.cpp )
target_link_libraries( jacobi PRIVATE ${DEPENDS_ON} )
install(TARGETS jacobi DESTINATION ${CMAKE_INSTALL_BINDIR})
