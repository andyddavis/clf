#ifndef CLF_PYBIND11_WRAPPERS_HPP_
#define CLF_PYBIND11_WRAPPERS_HPP_

#include <pybind11/pybind11.h>

namespace clf {
namespace python {

/// Implement the python interface for the clf::BasisFunctions class
/**
@param[in] mod The module that holds the python interface
*/
void BasisFunctionsWrapper(pybind11::module& mod);

/// Implement the python interface for the clf::SupportPoint class
/**
@param[in] mod The module that holds the python interface
*/
void SupportPointWrapper(pybind11::module& mod);

/// Implement the python interface for the clf::SupportPointCloud class
/**
@param[in] mod The module that holds the python interface
*/
void SupportPointCloudWrapper(pybind11::module& mod);

} // namespace python
} // namespace clf

#endif
