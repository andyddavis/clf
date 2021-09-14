#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/CoupledSupportPoint.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::CoupledSupportPointWrapper(pybind11::module& mod) {
  py::class_<CoupledSupportPoint, SupportPoint, std::shared_ptr<CoupledSupportPoint> > suppPt(mod, "CoupledSupportPoint");
  suppPt.def(py::init( [] (Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, py::dict const& d) { return CoupledSupportPoint::Construct(x, model, ConvertDictToPtree(d)); }));
}
