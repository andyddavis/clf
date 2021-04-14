macro(CLFDependency name)
    # try to find the package
    find_package(${name})

    # assume that we did not build this package
    set(CLF_BUILT_${name} OFF)

    if( ${${name}_FOUND} )
        # check that the package works
        include(Check${name})
        if( NOT ${${name}_COMPILES})
            # build and re-check the build
            include(Build${name})
            set(CLF_BUILT_${name} ON)
        endif()
    else()
        # build and check the build
        include(Build${name})
        set(CLF_BUILT_${name} ON)
    endif()

    # fail if we could not find this dependency
    if( NOT ${${name}_COMPILES} )
        message(FATAL_ERROR "\nCLF FAILED TO FIND OR BUILD ${name}\n")
    endif()

    # append the dependency information into the include directories and external libraries
    include(Append${name})
endmacro(CLFDependency)

# the prefix and suffix for library names
set(library_prefix "lib")
set(static_library_suffix ".a")
if( APPLE AND NOT CLF_BUILD_FROM_PIP )
  set(shared_library_suffix ".dylib")
else()
  set(shared_library_suffix ".so")
endif()

set(SPIPACK_BUILT_DEPENDENCIES )

CLFDependency(BOOST)
if( CLF_BUILT_BOOST )
  list(APPEND CLF_BUILT_DEPENDENCIES BOOST)
endif()

CLFDependency(EIGEN3)
if( CLF_BUILT_EIGEN3 )
  list(APPEND CLF_BUILT_DEPENDENCIES EIGEN3)
endif()

CLFDependency(MUQ)
if( CLF_BUILT_MUQ )
  list(APPEND CLF_BUILT_DEPENDENCIES MUQ)
endif()

if( NOT CLF_BUILD_FROM_PIP )
    CLFDependency(GTEST)
endif()

# add the header only submodules
list(APPEND CLF_EXTERNAL_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/external/nanoflann/include/
)
