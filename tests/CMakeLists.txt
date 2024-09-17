if(ENABLE_TESTS)
  blt_add_executable(NAME goltest
    SOURCES GOL.cpp
    #HEADERS tstDriver.hpp
    DEPENDS_ON Cabana::Core gtest)
  blt_add_test(NAME goltest
    COMMAND goltest)

  blt_add_executable(NAME jacobitest
    SOURCES Jacobi.cpp
    DEPENDS_ON gtest Cabana::Core)
  blt_add_test(NAME jacobitest
    COMMAND jacobitest)
  
endif()
