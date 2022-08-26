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
	    include(Check${name})
        endif()
    else()
        # build and check the build
        include(Build${name})
        set(CLF_BUILT_${name} ON)
    endif()

    # append the dependency information into the include directories and external libraries
    include(Append${name})
endmacro(CLFDependency)

# the prefix and suffix for library names
set(library_prefix "lib")
set(static_library_suffix ".a")
if( APPLE )
  set(shared_library_suffix ".dylib")
else()
  set(shared_library_suffix ".so")
endif()

set(CLF_BUILT_DEPENDENCIES 0)
set(CLF_DEPENDENCIES )

CLFDependency(EIGEN3)
if( CLF_BUILT_EIGEN3 )
  set(CLF_BUILT_DEPENDENCIES 1)
  list(APPEND CLF_DEPENDENCIES EIGEN3)
endif()

CLFDependency(GTEST)

