#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/BasisFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::BasisFunctionsWrapper(pybind11::module& mod) {
  py::class_<BasisFunctions, std::shared_ptr<BasisFunctions> > basis(mod, "BasisFunctions");
  basis.def("NumBasisFunctions", &BasisFunctions::NumBasisFunctions);
  basis.def("EvaluateBasisFunctions", &BasisFunctions::EvaluateBasisFunctions);
  basis.def("EvaluateBasisFunction", &BasisFunctions::EvaluateBasisFunction);
  basis.def("EvaluateBasisFunctionDerivative", &BasisFunctions::EvaluateBasisFunctionDerivative);
  basis.def("EvaluateBasisFunctionDerivatives", static_cast<Eigen::MatrixXd (BasisFunctions::*)(Eigen::VectorXd const& x, std::size_t const k) const>(&BasisFunctions::EvaluateBasisFunctionDerivatives));
  basis.def("EvaluateBasisFunctionDerivatives", static_cast<Eigen::VectorXd (BasisFunctions::*)(Eigen::VectorXd const& x, std::size_t const p, std::size_t const k) const>(&BasisFunctions::EvaluateBasisFunctionDerivatives));
  basis.def("FunctionEvaluation", &BasisFunctions::FunctionEvaluation);

  py::class_<PolynomialBasis, BasisFunctions, std::shared_ptr<PolynomialBasis> > poly(mod, "PolynomialBasis");
  poly.def_static("TotalOrderBasis", [](py::dict const& d) { return PolynomialBasis::TotalOrderBasis(ConvertDictToPtree(d)); });

  py::class_<SinCosBasis, BasisFunctions, std::shared_ptr<SinCosBasis> > sincos(mod, "SinCosBasis");
  sincos.def_static("TotalOrderBasis", [](py::dict const& d) { return SinCosBasis::TotalOrderBasis(ConvertDictToPtree(d)); });
}
