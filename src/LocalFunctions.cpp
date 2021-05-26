#include "clf/LocalFunctions.hpp"

#include <MUQ/Optimization/NLoptOptimizer.h>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

LocalFunctions::LocalFunctions(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
cloud(cloud),
globalCost(ConstructGlobalCost(cloud, pt)),
optimizationOptions(GetOptimizationOptions(pt))
{
  ComputeOptimalCoefficients();
}

pt::ptree LocalFunctions::GetOptimizationOptions(pt::ptree const& pt) {
  auto opt = pt.get_child_optional("Optimization");
  if( !opt ) { return pt::ptree(); }
  return opt.get();
}

double LocalFunctions::ComputeOptimalCoefficients() {
  if( globalCost ) { return ComputeCoupledSupportPoints(); }
  return ComputeIndependentSupportPoints();
}

double LocalFunctions::ComputeCoupledSupportPoints() {
  std::cout << std::endl << "COMPTUING COUPLED SUPPORT POINTS" << std::endl << std::endl;

  assert(globalCost);

  // get the current coefficients---they make up the initial guess
  Eigen::VectorXd coefficients = cloud->GetCoefficients();

  const double initCost = globalCost->Cost(coefficients);
  double prevCost = initCost;

  std::cout << "first prev cost: " << prevCost << std::endl;

  for( std::size_t iter=0; iter<optimizationOptions.maxEvals; ++iter ) {
    // compute the cost function gradient
    const Eigen::VectorXd grad = globalCost->Gradient(coefficients);

    // if the gradient is small, break
    if( grad.norm()<optimizationOptions.atol_grad ) { break; }

    // compute the Hessian
    Eigen::SparseMatrix<double> hess;
    globalCost->Hessian(coefficients, optimizationOptions.useGaussNewtonHessian, hess);
    assert(hess.rows()==coefficients.size());
    assert(hess.cols()==coefficients.size());

    // step direction xk-xkp1 = H^{-1} g
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<typename Eigen::SparseMatrix<double>::StorageIndex> > solver;
    solver.compute(hess);
    Eigen::VectorXd stepDir = solver.solve(grad);

    // do the line search
    double alpha = optimizationOptions.maxStepSizeScale;
    double newCost = globalCost->Cost((coefficients-alpha*stepDir).eval());
    std::size_t lineSearchIter = 0;
    while( newCost>prevCost || lineSearchIter++>optimizationOptions.maxLineSearchIterations ) {
      alpha *= optimizationOptions.lineSearchFactor;
      newCost = globalCost->Cost((coefficients-alpha*stepDir).eval());
    }

    // make a step
    if( alpha<1.0-1.0e-10 ) { stepDir *= alpha; }
    const double stepsize = stepDir.norm();
    prevCost = newCost;
    coefficients -= stepDir;

    std::cout << "prev cost: " << prevCost << std::endl;

    // check convergence
    if(
      prevCost<optimizationOptions.atol_function |
      prevCost/std::max(1.0e-10, initCost)<optimizationOptions.rtol_function |
      stepsize<optimizationOptions.atol_step |
      stepsize/std::max(1.0e-10, coefficients.norm())<optimizationOptions.rtol_step
    ) { break; }
  }

  // set the new coefficients
  cloud->SetCoefficients(coefficients);
  cost =  prevCost;

  return cost;
}

double LocalFunctions::ComputeIndependentSupportPoints() {
  std::cout << std::endl << "COMPTUING INDEPENDENT SUPPORT POINTS" << std::endl << std::endl;
  cost = 0.0;
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) { cost += (*point)->MinimizeUncoupledCost(); }
  cost /= cloud->NumSupportPoints();
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
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) {
    if( (*point)->couplingScale>0.0 ) { return std::make_shared<GlobalCost>(cloud, pt); }
  }
  return nullptr;
}
