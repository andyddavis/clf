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
if( NOT CLF_BUILD_FROM_PIP )
    set(CLF_BUILD_FROM_PIP 0)

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

if( CLF_BUILD_FROM_PIP )
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)
endif()

###############################################
# The source code
###############################################

# the source code (no pybind11)
set(CLF_SOURCE
    src/CLFException.cpp

    src/UtilityFunctions.cpp

    src/BasisFunctionsExceptions.cpp
    src/BasisFunctions.cpp
    src/SinCosBasis.cpp
    src/PolynomialBasis.cpp
    src/SupportPointBasis.cpp

    src/ModelExceptions.cpp
    src/Model.cpp

    src/UncoupledCost.cpp
    src/CoupledCost.cpp

    src/OptimizationOptions.cpp

    src/SupportPointExceptions.cpp
    src/SupportPoint.cpp

    src/SupportPointCloudExceptions.cpp
    src/SupportPointCloud.cpp

    src/LocalFunctions.cpp
)

# if we are building for python, include the pybind11
if( CLF_BUILD_FROM_PIP )
    set(CLF_SOURCE
        ${CLF_SOURCE}

        python/Module.cpp

        python/BasisFunctions.cpp

        python/Model.cpp

        python/SupportPoint.cpp

        python/SupportPointCloud.cpp

        python/LocalFunctions.cpp
    )
endif()

###############################################
# Build the CoupledLocalFunctions library
###############################################

set(LIBNAME CoupledLocalFunctions)

if( CLF_BUILD_FROM_PIP )
    pybind11_add_module(${LIBNAME} SHARED NO_EXTRAS ${CLF_SOURCE})
else()
    add_library(${LIBNAME} SHARED ${CLF_SOURCE})
endif()
target_link_libraries(${LIBNAME} PUBLIC ${CLF_EXTERNAL_LIBRARIES})
add_dependencies(${LIBNAME} ${CLF_BUILT_DEPENDENCIES})

###############################################
# Build gtests (if not a python install)
###############################################

if( NOT CLF_BUILD_FROM_PIP )
    set(TEST_SOURCE
        tests/_run-tests.cpp

        tests/BasisFunctionsTests.cpp
        tests/SinCosBasisTests.cpp
        tests/PolynomialBasisTests.cpp
        tests/SupportPointBasisTests.cpp

        tests/ModelTests.cpp

        tests/UncoupledCostTests.cpp
        tests/CoupledCostTests.cpp

        tests/SupportPointTests.cpp

        tests/SupportPointCloudTests.cpp

        tests/LocalFunctionsTests.cpp
    )

    include_directories(${GTEST_INCLUDE_DIR})

    add_executable(run-tests ${TEST_SOURCE})
    target_link_libraries(run-tests ${LIBNAME} ${CLF_EXTERNAL_LIBRARIES} ${GTEST_LIBRARIES})
    add_dependencies(run-tests ${LIBNAME})
    if( CLF_BUILT_GTEST )
        add_dependencies(run-tests GTEST)
    endif()
endif()

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
endif(DOXYGEN_FOUND)
