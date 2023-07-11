project(Manicore)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

# Dimension to compile for
ADD_DEFINITIONS(-DMAX_DIMENSION=2)
#------------------------------------------------------------------------------#
#                             Compiler setup
#------------------------------------------------------------------------------#

SET(CMAKE_CXX_COMPILER "g++")
#SET(CMAKE_CXX_COMPILER "clang++")
SET(CMAKE_CXX_STD "-std=c++20")
SET(CMAKE_CXX_FLAGS "-g -Wall -Wpedantic -Werror -Wno-uninitialized")
SET(CMAKE_CXX_FLAGS_DEBUGALL "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fsanitize=address,undefined -pthread" )
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -O3 -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Debugall)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type : ${CMAKE_BUILD_TYPE}")

OPTION(BUILD_TEST "Build the testing tree." ON)
#------------------------------------------------------------------------------#
#                             Required libraries
#------------------------------------------------------------------------------#

FIND_PACKAGE(Eigen3 3.4 REQUIRED)
SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${EXTERNAL_INCLUDES})
INCLUDE_DIRECTORIES(external)

#------------------------------------------------------------------------------#
#                             Directories for header libraries
#------------------------------------------------------------------------------#
INCLUDE_DIRECTORIES(include)
#------------------------------------------------------------------------------#
#                             Directories for compiled libraries
#------------------------------------------------------------------------------#

INCLUDE_DIRECTORIES(src/Mesh)
INCLUDE_DIRECTORIES(src/Loader)
INCLUDE_DIRECTORIES(src/Quadrature)
INCLUDE_DIRECTORIES(src/Integration)
INCLUDE_DIRECTORIES(src/DDR)

ADD_SUBDIRECTORY(src/Mesh)
ADD_SUBDIRECTORY(src/Loader)
ADD_SUBDIRECTORY(src/Quadrature)
ADD_SUBDIRECTORY(src/Integration)
ADD_SUBDIRECTORY(src/DDR)

#------------------------------------------------------------------------------#
#                             Directories for test
#------------------------------------------------------------------------------#

IF(BUILD_TEST)
  ADD_SUBDIRECTORY(test)
ENDIF()

#------------------------------------------------------------------------------#
#                             Directories for schemes
#------------------------------------------------------------------------------#

INCLUDE_DIRECTORIES(Schemes)
ADD_SUBDIRECTORY(Schemes)

#------------------------------------------------------------------------------#
#                             Directories for compiled meshes mappings
#------------------------------------------------------------------------------#

ADD_SUBDIRECTORY(meshes)

