set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo 
  MPI::MPI_CXX
  )
       
add_executable( cabanagol Main.cpp )
target_link_libraries( cabanagol PRIVATE ${DEPENDS_ON} )

install(TARGETS cabanagol DESTINATION ${CMAKE_INSTALL_LIBDIR})
