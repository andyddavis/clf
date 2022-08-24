#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/ConservationLaw.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::ConservationLawWrapper(py::module& mod) {
  py::class_<ConservationLaw, std::shared_ptr<ConservationLaw>, SystemOfEquations> sys(mod, "ConservationLaw");

  sys.def("Flux", &ConservationLaw::Flux);
  sys.def("Flux_JacobianWRTCoefficients", &ConservationLaw::Flux_JacobianWRTCoefficients);
  sys.def("Flux_JacobianWRTCoefficientsFD", &ConservationLaw::Flux_JacobianWRTCoefficientsFD);
  sys.def("Flux_HessianWRTCoefficients", &ConservationLaw::Flux_HessianWRTCoefficients);
  sys.def("Flux_HessianWRTCoefficientsFD", &ConservationLaw::Flux_HessianWRTCoefficientsFD);
  sys.def("FluxDivergence", &ConservationLaw::FluxDivergence);
  sys.def("FluxDivergenceFD", &ConservationLaw::FluxDivergenceFD);
  sys.def("FluxDivergence_GradientWRTCoefficients", &ConservationLaw::FluxDivergence_GradientWRTCoefficients);
  sys.def("FluxDivergence_GradientWRTCoefficientsFD", &ConservationLaw::FluxDivergence_GradientWRTCoefficientsFD);
  sys.def("FluxDivergence_HessianWRTCoefficients", &ConservationLaw::FluxDivergence_HessianWRTCoefficients);
  sys.def("FluxDivergence_HessianWRTCoefficientsFD", &ConservationLaw::FluxDivergence_HessianWRTCoefficientsFD);


}
