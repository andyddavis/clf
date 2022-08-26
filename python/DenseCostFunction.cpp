#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/DenseCostFunction.hpp"

namespace py = pybind11;
using namespace clf;
  
void clf::python::DenseCostFunctionWrapper(pybind11::module& mod) {
  py::class_<DenseCostFunction, std::shared_ptr<DenseCostFunction>, CostFunction<Eigen::MatrixXd> > dense(mod, "DenseCostFunction");
  dense.def(py::init<DensePenaltyFunctions const&>());
  dense.def(py::init<std::shared_ptr<DensePenaltyFunction> const&>());
}
