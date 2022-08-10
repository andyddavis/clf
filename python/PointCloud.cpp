#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/PointCloud.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::PointCloudWrapper(py::module& mod) {
  py::class_<PointCloud, std::shared_ptr<PointCloud> > cloud(mod, "PointCloud");
  cloud.def(py::init<>());

  cloud.def("AddPoint", static_cast<void(PointCloud::*)(Point const&)>(&PointCloud::AddPoint));
  cloud.def("AddPoint", static_cast<void(PointCloud::*)(Eigen::VectorXd const&)>(&PointCloud::AddPoint));
  cloud.def("NumPoints", &PointCloud::NumPoints);
  cloud.def("Get", &PointCloud::Get);
}
