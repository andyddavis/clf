#include "clf/python/Pybind11Wrappers.hpp"

#include "clf/LocalResidual.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::LocalResidualWrapper(py::module& mod) {
  py::class_<LocalResidual, std::shared_ptr<LocalResidual>, Residual, DensePenaltyFunction> resid(mod, "LocalResidual");
  resid.def(py::init<std::shared_ptr<LocalFunction> const&, std::shared_ptr<SystemOfEquations> const&,std::shared_ptr<const Parameters> const&>());
}
