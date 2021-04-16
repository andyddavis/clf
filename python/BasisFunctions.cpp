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
  basis.def("EvaluateBasisFunction", &BasisFunctions::EvaluateBasisFunction);
  basis.def("EvaluateBasisFunctions", &BasisFunctions::EvaluateBasisFunctions);

  py::class_<PolynomialBasis, BasisFunctions, std::shared_ptr<PolynomialBasis> > poly(mod, "PolynomialBasis");
  poly.def_static("TotalOrderBasis", [](py::dict const& d) { return PolynomialBasis::TotalOrderBasis(ConvertDictToPtree(d)); });

  py::class_<SinCosBasis, BasisFunctions, std::shared_ptr<SinCosBasis> > sincos(mod, "SinCosBasis");
  sincos.def_static("TotalOrderBasis", [](py::dict const& d) { return SinCosBasis::TotalOrderBasis(ConvertDictToPtree(d)); });
}
