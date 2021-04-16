#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/SupportPointCloud.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

namespace clf {
/// A support point cloud for the python wrapper
class PySupportPointCloud : public SupportPointCloud {
public:

  /**
  @param[in] supportPoints The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  @param[in] pt The options for the support point cloud
  */
  inline PySupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, boost::property_tree::ptree const& pt) : SupportPointCloud(supportPoints, pt) {}

  virtual ~PySupportPointCloud() = default;

  /// Find the \f$k\f$ nearest neighbors
  /**
  @param[in] point We want to find the nearest neighbors of this point
  @param[in] k We want to find this many nearest neighbors
  \return First: The indices of the nearest neighbors, Second: The squared distances from the input point to its nearest neighbors
  */
  inline std::pair<std::vector<std::size_t>, std::vector<double> > FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k) const {
    std::pair<std::vector<std::size_t>, std::vector<double> > result;
    SupportPointCloud::FindNearestNeighbors(point, k, result.first, result.second);
    return result;
  }

private:
};
} // namespace clf

void clf::python::SupportPointCloudWrapper(pybind11::module& mod) {
  py::class_<PySupportPointCloud, std::shared_ptr<PySupportPointCloud> > suppPtCloud(mod, "SupportPointCloud");
  suppPtCloud.def(py::init( [] (std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, py::dict const& d) { return new PySupportPointCloud(supportPoints, ConvertDictToPtree(d)); }));
  suppPtCloud.def("NumSupportPoints", &SupportPointCloud::NumSupportPoints);
  suppPtCloud.def("GetSupportPoint", &SupportPointCloud::GetSupportPoint);
  suppPtCloud.def("InputDimension", &SupportPointCloud::InputDimension);
  suppPtCloud.def("OutputDimension", &SupportPointCloud::OutputDimension);
  suppPtCloud.def("FindNearestNeighbors", &PySupportPointCloud::FindNearestNeighbors);
}
