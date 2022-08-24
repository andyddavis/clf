gg#include "clf/CoupledLocalFunctions.hpp"

using namespace clf;

CoupledLocalFunctions::CoupledLocalFunctions(std::shared_ptr<PointCloud> const& cloud) :
  cloud(cloud)
{}

std::size_t CoupledLocalFunctions::NumLocalFunctions() const { return cloud->NumPoints(); }
