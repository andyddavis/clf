#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

namespace py = pybind11;
              
PYBIND11_MODULE(PyCoupledLocalFunctions, module) {
  // a helper type for the sparse matrix interface
  py::class_<Eigen::Triplet<double> > triplet(module, "SparseEntry");
  triplet.def(py::init<std::size_t const, std::size_t const, double const>());

  clf::python::ParametersWrapper(module);
  
  clf::python::MultiIndexWrapper(module);
  clf::python::MultiIndexSetWrapper(module);
  
  clf::python::BasisFunctionsWrapper(module);
  clf::python::OrthogonalPolynomialsWrapper(module);

  clf::python::FeatureVectorWrapper(module);
  clf::python::FeatureMatrixWrapper(module);

  clf::python::LocalFunctionWrapper(module);

  clf::python::SystemOfEquationsWrapper(module);
  clf::python::IdentityModelWrapper(module);
  clf::python::LinearModelWrapper(module);

  clf::python::PenaltyFunctionWrapper(module);
  clf::python::CostFunctionWrapper(module);

  clf::python::PointWrapper(module);

  clf::python::LocalResidualWrapper(module);
}
