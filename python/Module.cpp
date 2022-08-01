#include "clf/python/Pybind11Wrappers.hpp"

namespace py = pybind11;
              
PYBIND11_MODULE(PyCoupledLocalFunctions, module) {
  clf::python::ParametersWrapper(module);
  
  clf::python::MultiIndexWrapper(module);
  clf::python::MultiIndexSetWrapper(module);
  
  clf::python::BasisFunctionsWrapper(module);
  clf::python::OrthogonalPolynomialsWrapper(module);

  clf::python::FeatureVectorWrapper(module);
  clf::python::FeatureMatrixWrapper(module);

  clf::python::LocalFunctionWrapper(module);
}
