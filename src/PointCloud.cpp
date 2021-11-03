#include "clf/PointCloud.hpp"

#include <MUQ/Utilities/HDF5/HDF5File.h>

using namespace muq::Utilities;
using namespace clf;

PointCloud::PointCloud() {}

PointCloud::PointCloud(std::vector<std::shared_ptr<Point> > const& points) :
points(points)
{}

std::vector<std::shared_ptr<Point> >::const_iterator PointCloud::Begin() const { return points.begin(); }

std::vector<std::shared_ptr<Point> >::const_iterator PointCloud::End() const { return points.end(); }

std::size_t PointCloud::NumPoints() const { return points.size(); }

void PointCloud::WriteToFile(std::string const& filename, std::string const& dataset, std::string const& dataname) const {
  if( points.size()==0 ) { return; }

  assert(filename.size()>3);
  assert(filename.substr(filename.size()-3)==".h5");
  assert(dataset.size()>=1);
  assert(dataset.substr(0)=="/");
  assert(dataname.size()>=1);

  Eigen::MatrixXd pnts(points.size(), points[0]->x.size());
  for( std::size_t i=0; i<points.size(); ++i ) { pnts.row(i) = points[i]->x; }

  HDF5File file(filename);
  file.WriteMatrix(dataset+dataname, pnts);
  file.Close();
}

std::shared_ptr<Point> PointCloud::GetPoint(std::size_t const ind) const {
  assert(ind<points.size());
  return points[ind];
}
