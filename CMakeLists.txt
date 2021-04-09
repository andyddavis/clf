cmake_minimum_required (VERSION 3.0)

project(CLF)

find_package(MUQ HINTS ~/Software/install/muq2/CMake/MUQ REQUIRED)

find_package(Trilinos HINTS ~/Software/install/trilinos/ REQUIRED)

set(GTEST_DIR ~/Software/install/googletest/)

find_path(GTEST_INCLUDE_DIR gtest/gtest.h
         HINTS ${GTEST_DIR}/include ${GTEST_DIR}/include
         PATH_SUFFIXES gtest NO_DEFAULT_PATH)

find_library(GTEST_LIBRARIES NAMES libgtest.a
             HINTS ${GTEST_DIR}/lib ${GTEST_DIR}/build NO_DEFAULT_PATH)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${MUQ_CXX_FLAGS} -fopenmp -O3")
set(CMAKE_CXX_COMPILER ${MUQ_CXX_COMPILER})

include_directories(${MUQ_INCLUDE_DIRS} ${Trilinos_INCLUDE_DIRS} ${GTEST_INCLUDE_DIR} .)

set(CLF_SOURCE
    src/BasisFunctions.cpp

    src/SupportPoint.cpp
)

add_library(CLF SHARED ${CLF_SOURCE})
target_link_libraries(CLF ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES} ${Trilinos_LIBRARIES} ${GTEST_LIBRARIES})

set(TEST_SOURCE
    tests/_run-tests.cpp

    tests/SupportPointTests.cpp
)

add_executable(run-tests ${TEST_SOURCE})
target_link_libraries(run-tests CLF ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES} ${Trilinos_LIBRARIES} ${GTEST_LIBRARIES})
add_dependencies(run-tests CLF)

###############################################
# Build the documentation (doxygen)
###############################################

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
	message(STATUS "To build documentation, run 'make doc'")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/documentation/clf.doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/clf.doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/clf.doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
