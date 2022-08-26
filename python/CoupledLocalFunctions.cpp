#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "clf/CoupledLocalFunctions.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::CoupledLocalFunctionsWrapper(py::module& mod) {
  py::class_<CoupledLocalFunctions, std::shared_ptr<CoupledLocalFunctions> > func(mod, "CoupledLocalFunctions");
  func.def(py::init<std::shared_ptr<MultiIndexSet> const&, std::shared_ptr<BasisFunctions> const&, std::shared_ptr<Domain> const&, Eigen::VectorXd const&, std::shared_ptr<Parameters> const&>());

  func.def("NumLocalFunctions", &CoupledLocalFunctions::NumLocalFunctions);
  func.def("AddBoundaryCondition", &CoupledLocalFunctions::AddBoundaryCondition);
  func.def("GetResiduals", &CoupledLocalFunctions::GetResiduals);
  func.def("RemoveResidual", &CoupledLocalFunctions::RemoveResidual);
  func.def("AddResidual", &CoupledLocalFunctions::AddResidual);
}
