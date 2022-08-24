#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/LocalFunction.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::FeatureVectorWrapper(py::module& mod) {
  py::class_<FeatureVector, std::shared_ptr<FeatureVector> > vec(mod, "FeatureVector");
  vec.def(py::init<std::shared_ptr<MultiIndexSet> const&, std::shared_ptr<BasisFunctions> const&, std::shared_ptr<Domain> const&>());
  vec.def(py::init<std::shared_ptr<MultiIndexSet> const&, std::shared_ptr<BasisFunctions> const&>());

  vec.def("InputDimension", &FeatureVector::InputDimension);
  vec.def("NumBasisFunctions", &FeatureVector::NumBasisFunctions);
  vec.def("Evaluate", &FeatureVector::Evaluate);
  vec.def("Derivative", static_cast<Eigen::MatrixXd (FeatureVector::*)(Eigen::VectorXd const&, Eigen::MatrixXi const&) const>(&FeatureVector::Derivative));
  vec.def("Derivative", static_cast<Eigen::MatrixXd (FeatureVector::*)(Eigen::VectorXd const&, Eigen::MatrixXi const&, std::optional<Eigen::VectorXd> const&) const>(&FeatureVector::Derivative));
}
