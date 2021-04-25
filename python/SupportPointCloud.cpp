#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/SupportPointCloud.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::SupportPointCloudWrapper(pybind11::module& mod) {
  py::class_<SupportPointCloud, std::shared_ptr<SupportPointCloud> > suppPtCloud(mod, "SupportPointCloud");
  suppPtCloud.def(py::init( [] (std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, py::dict const& d) { return SupportPointCloud::Construct(supportPoints, ConvertDictToPtree(d)); }));
  suppPtCloud.def("NumSupportPoints", &SupportPointCloud::NumSupportPoints);
  suppPtCloud.def("GetSupportPoint", &SupportPointCloud::GetSupportPoint);
  suppPtCloud.def("InputDimension", &SupportPointCloud::InputDimension);
  suppPtCloud.def("OutputDimension", &SupportPointCloud::OutputDimension);
  suppPtCloud.def("FindNearestNeighbors", static_cast<std::pair<std::vector<std::size_t>, std::vector<double> > (SupportPointCloud::*)(Eigen::VectorXd const& point, std::size_t const k) const>(&SupportPointCloud::FindNearestNeighbors));
}
