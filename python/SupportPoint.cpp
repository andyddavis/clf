#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/SupportPoint.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::SupportPointWrapper(pybind11::module& mod) {
  py::class_<SupportPoint, std::shared_ptr<SupportPoint> > suppPt(mod, "SupportPoint");
  suppPt.def(py::init( [] (Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, py::dict const& d) { return SupportPoint::Construct(x, model, ConvertDictToPtree(d)); }));
  suppPt.def_readonly("x", &SupportPoint::x);
  suppPt.def_readonly("model", &SupportPoint::model);
  suppPt.def("GetBasisFunctions", &SupportPoint::GetBasisFunctions);
  suppPt.def("NumNeighbors", &SupportPoint::NumNeighbors);
  suppPt.def("NumCoefficients", &SupportPoint::NumCoefficients);
  suppPt.def("EvaluateLocalFunction", static_cast<Eigen::VectorXd (SupportPoint::*)(Eigen::VectorXd const& loc) const>(&SupportPoint::EvaluateLocalFunction));
  suppPt.def("EvaluateLocalFunction", static_cast<Eigen::VectorXd (SupportPoint::*)(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const>(&SupportPoint::EvaluateLocalFunction));
}
