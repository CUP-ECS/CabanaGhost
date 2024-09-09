if(ENABLE_TESTS)
  blt_add_executable(NAME goltest2
    SOURCES GOL.cpp
    DEPENDS_ON Cabana::Core gtest)
  blt_add_test(NAME goltest2
    COMMAND goltest2)

  blt_add_executable(NAME jacobitest2
    SOURCES Jacobi.cpp
    DEPENDS_ON gtest Cabana::Core)
  blt_add_test(NAME jacobitest2
    COMMAND jacobitest2)
  
endif()
