find_package(PkgConfig)

set(BOOST_MIN_VERSION "1.56.0")

if(NOT DEFINED CLF_BOOST_DIR)
	unset(Boost_LIBRARIES)
	unset(Boost_INCLUDE_DIR)
	unset(Boost_LIBRARY_DIRS)
	set(Boost_USE_STATIC_LIBS ON)

	find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS system filesystem graph)

	IF(Boost_FOUND)
		set(BOOST_LIBRARIES_STATIC ${Boost_LIBRARIES})
	endif()

	unset(Boost_LIBRARIES)
	unset(Boost_INCLUDE_DIR)
	unset(Boost_LIBRARY_DIRS)
	unset(Boost_USE_STATIC_LIBS)

	find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS system filesystem graph)

	IF(Boost_FOUND)
		set(BOOST_LIBRARIES ${Boost_LIBRARIES})
		set(BOOST_INCLUDE_DIR ${Boost_INCLUDE_DIR})
	endif()

else()
	find_path(BOOST_INCLUDE_DIR boost/property_tree/ptree.hpp
	          HINTS ${CLF_BOOST_DIR}/include ${CLF_BOOST_DIR} ${CLF_BOOST_INCLUDE_DIR}
	          PATH_SUFFIXES boost NO_DEFAULT_PATH)

 	find_library(BOOST_SYSTEM_LIBRARY NAMES boost_system
 		 	     HINTS ${CLF_BOOST_DIR} ${CLF_BOOST_DIR}/lib ${CLF_BOOST_DIR}/stage/lib NO_DEFAULT_PATH)
 	find_library(BOOST_FILESYSTEM_LIBRARY NAMES boost_filesystem
 		 	     HINTS ${CLF_BOOST_DIR} ${CLF_BOOST_DIR}/lib ${CLF_BOOST_DIR}/stage/lib NO_DEFAULT_PATH)
 	find_library(BOOST_GRAPH_LIBRARY NAMES boost_graph
 		 	     HINTS ${CLF_BOOST_DIR} ${CLF_BOOST_DIR}/lib ${CLF_BOOST_DIR}/stage/lib NO_DEFAULT_PATH)

	set(BOOST_LIBRARY ${BOOST_GRAPH_LIBRARY} ${BOOST_FILESYSTEM_LIBRARY} ${BOOST_SYSTEM_LIBRARY})
	set(BOOST_LIBRARIES ${BOOST_LIBRARY})
endif()

set(BOOST_INCLUDE_DIRS ${BOOST_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(BOOST  DEFAULT_MSG
                                  BOOST_LIBRARY BOOST_INCLUDE_DIR)

mark_as_advanced(BOOST_INCLUDE_DIR BOOST_LIBRARY)
