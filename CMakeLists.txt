cmake_minimum_required(VERSION 3.4...3.18)
project(CoupledLocalFunctions)

# opening message
message(STATUS "")
message(STATUS "Welcome to the Coupled Local Functions (CLF) library!")
message(STATUS "")

###############################################
# Check flags
###############################################

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-w -g -pthread -fopenmp -O3 -ldl -fPIC")

# check if this is a python build or not
if( NOT CLF_USE_PYTHON )
    set(CLF_USE_PYTHON 1)

    # if not a python build, check to see where to be third party installs---if not set by the user, put it into the install directory
    if( NOT CLF_EXTERNAL_INSTALL_DIR )
        set(CLF_EXTERNAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/clf_external)
    endif()
endif()

###############################################
# Find dependencies
###############################################

# tell cmake where to find the cmake source code
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/
	${CMAKE_SOURCE_DIR}/cmake/find/
	${CMAKE_SOURCE_DIR}/cmake/check/
	${CMAKE_SOURCE_DIR}/cmake/build/
	${CMAKE_SOURCE_DIR}/cmake/append/
	)

# find dependencies
include(FindDependencies)

# print external libraries
message(STATUS "")
message(STATUS "External libraries: " ${CLF_EXTERNAL_LIBRARIES})
message(STATUS "External includes: " ${CLF_EXTERNAL_INCLUDE_DIRS})
message(STATUS "")

include_directories(${CLF_EXTERNAL_INCLUDE_DIRS} .)

if( CLF_USE_PYTHON )
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)
endif()

###############################################
# The source code
###############################################

# the source code 
set(CLF_SOURCE
    src/CLFExceptions.cpp

    src/Parameters.cpp

    src/FiniteDifference.cpp

    src/PenaltyFunction.cpp
    src/CostFunction.cpp

    src/BasisFunctions.cpp
    src/OrthogonalPolynomials.cpp
    src/LegendrePolynomials.cpp

    src/FeatureVector.cpp
    src/FeatureMatrix.cpp

    src/MultiIndex.cpp
    src/MultiIndexSet.cpp

    src/LocalFunction.cpp

    src/SystemOfEquations.cpp
    src/IdentityModel.cpp
    src/LinearModel.cpp

    src/LevenbergMarquardt.cpp

    src/LocalResidual.cpp

    src/Point.cpp	

    src/PointCloud.cpp
)

# if we are building for python, include the pybind11
if( CLF_USE_PYTHON )
    set(CLF_SOURCE_PYTHON
	python/Module.cpp

	python/Parameters.cpp

	python/MultiIndex.cpp
	python/MultiIndexSet.cpp

	python/BasisFunctions.cpp
	python/OrthogonalPolynomials.cpp

	python/FeatureVector.cpp
	python/FeatureMatrix.cpp

	python/LocalFunction.cpp

	python/SystemOfEquations.cpp
	python/IdentityModel.cpp
	python/LinearModel.cpp

	python/PenaltyFunction.cpp
	python/CostFunction.cpp

	python/LevenbergMarquardt.cpp

	python/Point.cpp
	python/PointCloud.cpp
	
	python/LocalResidual.cpp
    )
endif()

###############################################
# Build the CoupledLocalFunctions library
###############################################

set(CLF_LIBNAME CoupledLocalFunctions)
set(CLF_PYTHON_LIBNAME PyCoupledLocalFunctions)

add_library(${CLF_LIBNAME} SHARED ${CLF_SOURCE})
target_link_libraries(${CLF_LIBNAME} PUBLIC ${CLF_EXTERNAL_LIBRARIES})
if( CLF_BUILT_DEPENDENCIES ) 
    add_dependencies(${CLF_LIBNAME} ${CLF_DEPENDENCIES})
endif()
if( CLF_USE_PYTHON )
  pybind11_add_module(${CLF_PYTHON_LIBNAME} SHARED NO_EXTRAS ${CLF_SOURCE_PYTHON})
  target_link_libraries(${CLF_PYTHON_LIBNAME} PUBLIC ${CLF_LIBNAME} ${CLF_EXTERNAL_LIBRARIES})
  add_dependencies(${CLF_PYTHON_LIBNAME} ${CLF_LIBNAME})
  if( CLF_BUILT_DEPENDENCIES ) 
    add_dependencies(${CLF_PYTHON_LIBNAME} ${CLF_DEPENDENCIES})
  endif()
  #set_target_properties(${CLF_PYTHON_LIBNAME}
  #  PROPERTIES 
  #  CXX_STANDARD 17
  #  CXX_STANDARD_REQUIRED ON
  #  SUFFIX ".cpython-310-x86_64-linux-gnu"
  #  )
endif()

###############################################
# Build gtests (if not a python install)
###############################################

set(TEST_SOURCE
    tests/_run-tests.cpp

    tests/ParametersTests.cpp

    tests/LinearSolverTests.cpp

    tests/OrthogonalPolynomialsTests.cpp

    tests/MultiIndexTests.cpp
    tests/MultiIndexSetTests.cpp

    tests/FeatureVectorTests.cpp
    tests/FeatureMatrixTests.cpp

    tests/LocalFunctionTests.cpp

    tests/SystemOfEquationsTests.cpp	
    tests/IdentityModelTests.cpp
    tests/LinearModelTests.cpp

    tests/PenaltyFunctionTests.cpp
    tests/CostFunctionTests.cpp
    tests/LevenbergMarquardtTests.cpp

    tests/LocalResidualTests.cpp

    tests/PointTests.cpp

    tests/PointCloudTests.cpp
    )

include_directories(tests/ ${GTEST_INCLUDE_DIR})

add_executable(run-tests ${TEST_SOURCE})
target_link_libraries(run-tests ${CLF_LIBNAME} ${CLF_EXTERNAL_LIBRARIES} ${GTEST_LIBRARIES})
add_dependencies(run-tests ${CLF_LIBNAME})
if( CLF_BUILT_GTEST )
    add_dependencies(run-tests GTEST)
endif()

###############################################
# Build examples
###############################################

#set(POISSON_1D_EXAMPLE_SOURCE
#    examples/Poisson-equation/cpp/_run-1d.cpp
#)

#add_executable(run-poisson-1d ${POISSON_1D_EXAMPLE_SOURCE})
#target_link_libraries(run-poisson-1d ${CLF_LIBNAME} ${CLF_EXTERNAL_LIBRARIES} ${GTEST_LIBRARIES})
#add_dependencies(run-poisson-1d ${CLF_LIBNAME})

###############################################
# Install step (if not a python install)
###############################################

# install all the CLF headers
install(DIRECTORY clf
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
  FILES_MATCHING PATTERN "*.hpp")

# install the library
install(TARGETS ${CLF_LIBNAME}
  EXPORT ${CMAKE_PROJECT_NAME}Depends
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

if( CLF_USE_PYTHON )
    install(TARGETS ${CLF_PYTHON_LIBNAME}
        EXPORT ${CMAKE_PROJECT_NAME}Depends
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Install the config file
configure_file(cmake/CLFConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${CMAKE_PROJECT_NAME}Config.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${CMAKE_PROJECT_NAME}Config.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}"
  COMPONENT dev)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR})

###############################################
# Build the documentation (doxygen)
###############################################

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if( DOXYGEN_FOUND AND NOT CLF_BUILD_FROM_PIP )
	message(STATUS "To build documentation, run 'make doc'")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/documentation/clf.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/clf.doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/clf.doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif()
