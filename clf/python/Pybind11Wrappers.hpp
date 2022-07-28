#ifndef PYBIND11WRAPPERS_HPP_
#define PYBIND11WRAPPERS_HPP_

#include <pybind11/pybind11.h>

namespace clf {
namespace python {

void ParametersWrapper(pybind11::module& mod);

} // namespace python 
} // namespace clf

#endif
