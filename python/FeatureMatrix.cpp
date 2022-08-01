#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/LocalFunction.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::FeatureMatrixWrapper(py::module& mod) {
  py::class_<FeatureMatrix, std::shared_ptr<FeatureMatrix> > mat(mod, "FeatureMatrix");
  mat.def(py::init<std::shared_ptr<const FeatureVector> const&, std::size_t const>());
  mat.def(py::init<std::vector<std::shared_ptr<const FeatureVector> > const&>());
  mat.def(py::init<std::vector<FeatureMatrix::VectorPair> const&>());

  mat.def("GetFeatureVector", &FeatureMatrix::GetFeatureVector);
  mat.def("InputDimension", &FeatureMatrix::InputDimension);
  mat.def("ApplyTranspose", &FeatureMatrix::ApplyTranspose);
  mat.def_readonly("numBasisFunctions", &FeatureMatrix::numBasisFunctions);
  mat.def_readonly("numFeatureVectors", &FeatureMatrix::numFeatureVectors);
}
