cmake_minimum_required(VERSION 3.4...3.18)
project(1d-example)

find_package(CoupledLocalFunctions REQUIRED HINTS ${CLF_DIR})

include_directories(${CLF_INCLUDE_DIRS})
link_directories(${CLF_LIBRARY_DIR})

set(SOURCE
    _run-1d-example.cpp
)

add_executable(run-1d-example ${SOURCE})
target_link_libraries(run-1d-example ${CLF_EXTERNAL_LIBRARIES} ${CLF_LIBRARY})
