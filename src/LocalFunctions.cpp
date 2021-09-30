#include "clf/LocalFunctions.hpp"

#include <MUQ/Optimization/NLoptOptimizer.h>

#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

LocalFunctions::LocalFunctions(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
cloud(cloud),
globalCost(ConstructGlobalCost(cloud, pt))
{}

double LocalFunctions::ComputeOptimalCoefficients(pt::ptree const& options) {
  if( globalCost ) { assert(false); return ComputeCoupledSupportPoints(); }
  return ComputeIndependentSupportPoints(options);
}

double LocalFunctions::ComputeOptimalCoefficients(Eigen::MatrixXd const& forcing, pt::ptree const& options) {
  if( globalCost ) { assert(false); return 0.0; }
  return ComputeIndependentSupportPoints(forcing, options);
}

double LocalFunctions::ComputeCoupledSupportPoints() {
  std::cout << "COMPUTE coupled support points" << std::endl;
  /*assert(globalCost);

  pt::ptree pt;
  auto lm = std::make_shared<SparseLevenbergMarquardt>(globalCost, pt);

  // get the current coefficients---they make up the initial guess
  Eigen::VectorXd coefficients = cloud->GetCoefficients();

  // compute the optimal coefficients
  Eigen::VectorXd costVec;
  lm->Minimize(coefficients, costVec);
  cost = costVec.dot(costVec);

  // set the new coefficients
  cloud->SetCoefficients(coefficients);

  return cost;*/
  return 0.0;
}

double LocalFunctions::ComputeIndependentSupportPoints(Eigen::MatrixXd const& forcing, boost::property_tree::ptree const& options) {
  cost = 0.0;

  #pragma omp parallel for num_threads(options.get<std::size_t>("NumThreads", 1))
  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    auto point = cloud->GetSupportPoint(i);
    assert(point);

    cost += point->MinimizeUncoupledCost(forcing, options);
  }
  cost /= cloud->NumPoints();
  return cost;
}

double LocalFunctions::ComputeIndependentSupportPoints(boost::property_tree::ptree const& options) {
  cost = 0.0;

  #pragma omp parallel for num_threads(options.get<std::size_t>("NumThreads", 1))
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    cost += point->MinimizeUncoupledCost(options);
  }
  cost /= cloud->NumPoints();
  return cost;
}

double LocalFunctions::CoefficientCost() const { return cost; }

Eigen::VectorXd LocalFunctions::Evaluate(Eigen::VectorXd const& x) const { return cloud->GetSupportPoint(NearestNeighborIndex(x))->EvaluateLocalFunction(x); }

std::size_t LocalFunctions::NearestNeighborIndex(Eigen::VectorXd const& x) const { return NearestNeighbor(x).first; }

double LocalFunctions::NearestNeighborDistance(Eigen::VectorXd const& x) const { return NearestNeighbor(x).second; }

std::pair<std::size_t, double> LocalFunctions::NearestNeighbor(Eigen::VectorXd const& x) const {
  // find the closest point to the input point
  std::vector<std::size_t> ind;
  std::vector<double> dist;
  cloud->FindNearestNeighbors(x, 1, ind, dist);
  return std::pair<std::size_t, double>(ind[0], dist[0]);
}

std::shared_ptr<GlobalCost> LocalFunctions::ConstructGlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt) {
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    if( point->Coupled() ) { return std::make_shared<GlobalCost>(cloud, pt); }
  }
  return nullptr;
}
