#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/SparseCostFunction.hpp"

namespace py = pybind11;
using namespace clf;
  
void clf::python::SparseCostFunctionWrapper(pybind11::module& mod) {
  py::class_<SparseCostFunction, std::shared_ptr<SparseCostFunction>, CostFunction<Eigen::SparseMatrix<double> > > sparse(mod, "SparseCostFunction");
  sparse.def(py::init<SparsePenaltyFunctions const&>());
  sparse.def(py::init<std::shared_ptr<SparsePenaltyFunction> const&>());
}
