#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/SupportPoint.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::SupportPointWrapper(pybind11::module& mod) {
  py::class_<SupportPoint, std::shared_ptr<SupportPoint> > suppPt(mod, "SupportPoint");
  suppPt.def(py::init( [] (Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, py::dict const& d) { return new SupportPoint(x, model, ConvertDictToPtree(d)); }));
  suppPt.def_readonly("x", &SupportPoint::x);
  suppPt.def_readonly("model", &SupportPoint::model);
  suppPt.def_readonly("bases", &SupportPoint::bases);
  suppPt.def_readonly("numNeighbors", &SupportPoint::numNeighbors);
}
