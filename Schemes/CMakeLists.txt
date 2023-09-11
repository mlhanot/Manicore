FIND_PACKAGE(Boost COMPONENTS program_options)

IF(Boost_FOUND)
  ADD_SUBDIRECTORY(Maxwell2D)
ELSE()
  MESSAGE(STATUS "Boost not found, skipping the compilation of some Schemes")
ENDIF()

