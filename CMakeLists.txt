cmake_minimum_required (VERSION 3.0)

project(CLF)

find_package(MUQ HINTS ~/Software/install/muq2/CMake/MUQ REQUIRED)
find_package(Trilinos HINTS ~/Software/install/trilinos/ REQUIRED)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${MUQ_CXX_FLAGS} -fopenmp -O3")
set(CMAKE_CXX_COMPILER ${MUQ_CXX_COMPILER})

set(TEST_SOURCE
    tests/_run-tests.cpp
)

add_executable(run-tests ${TEST_SOURCE})
target_link_libraries(run-tests ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES} ${Trilinos_LIBRARIES})
