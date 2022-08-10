#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/LocalFunction.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::LocalFunctionWrapper(py::module& mod) {
  py::class_<LocalFunction, std::shared_ptr<LocalFunction> > func(mod, "LocalFunction");
  func.def(py::init<std::shared_ptr<MultiIndexSet> const&, std::shared_ptr<BasisFunctions> const&, std::shared_ptr<Domain> const&, std::size_t const>());
  func.def(py::init<std::shared_ptr<MultiIndexSet> const&, std::shared_ptr<BasisFunctions> const&, std::shared_ptr<Domain> const&, std::shared_ptr<Parameters> const&>());

  func.def("InputDimension", &LocalFunction::InputDimension);
  func.def("OutputDimension", &LocalFunction::OutputDimension);
  func.def("NumCoefficients", &LocalFunction::NumCoefficients);
  func.def("SampleDomain", &LocalFunction::SampleDomain);
  func.def("Evaluate", static_cast<Eigen::VectorXd(LocalFunction::*)(Eigen::VectorXd const&, Eigen::VectorXd const&) const>(&LocalFunction::Evaluate));
  func.def_readonly("featureMatrix", &LocalFunction::featureMatrix);
}
