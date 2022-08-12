#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/PointCloud.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::PointCloudWrapper(py::module& mod) {
  py::class_<PointCloud, std::shared_ptr<PointCloud> > cloud(mod, "PointCloud");
  cloud.def(py::init<>());
  cloud.def(py::init<std::shared_ptr<Domain> const&>());

  cloud.def("AddPoint", static_cast<void(PointCloud::*)(std::shared_ptr<Point> const&)>(&PointCloud::AddPoint));
  cloud.def("AddPoint", static_cast<void(PointCloud::*)()>(&PointCloud::AddPoint));
  cloud.def("AddPoints", &PointCloud::AddPoints);
  cloud.def("NumPoints", &PointCloud::NumPoints);
  cloud.def("Get", &PointCloud::Get);
  cloud.def("Distance", &PointCloud::Distance);
  cloud.def("NearestNeighbors", &PointCloud::NearestNeighbors);
  cloud.def("ClosestPoint", &PointCloud::ClosestPoint);
}
