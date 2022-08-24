#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/ConservationLawWeakFormResidual.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::ConservationLawWeakFormResidualWrapper(pybind11::module& mod) {
  py::class_<ConservationLawWeakFormResidual, std::shared_ptr<ConservationLawWeakFormResidual>, DensePenaltyFunction> resid(mod, "ConservationLawWeakFormResidual");
  resid.def(py::init<std::shared_ptr<LocalFunction> const&, std::shared_ptr<ConservationLaw> const&, std::shared_ptr<FeatureVector> const&, std::shared_ptr<const Parameters> const&>());

  resid.def("NumBoundaryPoints", &ConservationLawWeakFormResidual::NumBoundaryPoints);
  resid.def("NumPoints", &ConservationLawWeakFormResidual::NumPoints);
  resid.def("GetBoundaryPoint", &ConservationLawWeakFormResidual::GetBoundaryPoint);
  resid.def("GetPoint", &ConservationLawWeakFormResidual::GetPoint);
}
