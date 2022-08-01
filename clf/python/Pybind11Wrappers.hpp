#ifndef PYBIND11WRAPPERS_HPP_
#define PYBIND11WRAPPERS_HPP_

#include <pybind11/pybind11.h>

namespace clf {
namespace python {

/// Python wrapper for clf::Parameters
/**
   @param[in] mod The python module
 */
void ParametersWrapper(pybind11::module& mod);

/// Python wrapper for clf::MultiIndex
/**
   @param[in] mod The python module
 */
void MultiIndexWrapper(pybind11::module& mod);

/// Python wrapper for clf::MultiIndexSet
/**
   @param[in] mod The python module
 */
void MultiIndexSetWrapper(pybind11::module& mod);

/// Python wrapper for clf::BasisFunctions
/**
   @param[in] mod The python module
 */
void BasisFunctionsWrapper(pybind11::module& mod);

/// Python wrapper for clf::OrthogonalPolynomials and its children
/**
   @param[in] mod The python module
 */
void OrthogonalPolynomialsWrapper(pybind11::module& mod);

/// Python wrapper for clf::FeatureVector
/**
   @param[in] mod The python module
 */
void FeatureVectorWrapper(pybind11::module& mod);

/// Python wrapper for clf::FeatureMatrix
/**
   @param[in] mod The python module
 */
void FeatureMatrixWrapper(pybind11::module& mod);

/// Python wrapper for clf::LocalFunction
/**
   @param[in] mod The python module
 */
void LocalFunctionWrapper(pybind11::module& mod);

} // namespace python 
} // namespace clf

#endif
