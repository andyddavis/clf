#ifndef PYBIND11WRAPPERS_HPP_
#define PYBIND11WRAPPERS_HPP_

#include <pybind11/pybind11.h>

namespace clf {
namespace python {

/// Python wrapper for clf::Parameters
/**
   @param[in] mod The python module
 */
void ParametersWrapper(pybind11::module& mod);

/// Python wrapper for clf::MultiIndex
/**
   @param[in] mod The python module
 */
void MultiIndexWrapper(pybind11::module& mod);

/// Python wrapper for clf::MultiIndexSet
/**
   @param[in] mod The python module
 */
void MultiIndexSetWrapper(pybind11::module& mod);

} // namespace python 
} // namespace clf

#endif
