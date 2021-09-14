#include "clf/LocalFunctions.hpp"

#include <MUQ/Optimization/NLoptOptimizer.h>

#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

LocalFunctions::LocalFunctions(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
cloud(cloud),
globalCost(ConstructGlobalCost(cloud, pt)),
optimizationOptions(GetOptimizationOptions(pt))
{
  std::cout << "GLOBAL COST: " << ComputeOptimalCoefficients() << std::endl;
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

Eigen::VectorXd LocalFunctions::StepDirection(Eigen::VectorXd const& coefficients, Eigen::VectorXd const& grad, double const lambda, bool const useGN) const {
  Eigen::SparseMatrix<double> hess;
  //globalCost->Hessian(coefficients, useGN, hess);
  assert(hess.rows()==coefficients.size()); assert(hess.cols()==coefficients.size());

  // add the scaled identity
  for( std::size_t i=0; i<coefficients.size(); ++i ) { hess.coeffRef(i, i) += lambda; }

  // step direction xk-xkp1 = H^{-1} g
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<typename Eigen::SparseMatrix<double>::StorageIndex> > solver;
  solver.compute(hess);
  return solver.solve(grad);
}

std::pair<double, double> LocalFunctions::LineSearch(Eigen::VectorXd const& coefficients, Eigen::VectorXd const& stepDir, double const prevCost) const {
  double alpha = optimizationOptions.maxStepSizeScale;
  //double newCost = globalCost->Cost((coefficients-alpha*stepDir).eval());
  double newCost = 0.0;
  std::size_t lineSearchIter = 0;
  while( newCost>prevCost || lineSearchIter++>optimizationOptions.maxLineSearchIterations ) {
    alpha *= optimizationOptions.lineSearchFactor;
    //newCost = globalCost->Cost((coefficients-alpha*stepDir).eval());
  }

  return std::pair<double, double>(alpha, newCost);
}

double LocalFunctions::ComputeCoupledSupportPoints() {
  assert(globalCost);

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

  return cost;
  /*std::cout << "STARTING" << std::endl;
  pt::ptree options;
  options.put("Ftol.AbsoluteTolerance", optimizationOptions.atol_function);
  options.put("Ftol.RelativeTolerance", optimizationOptions.rtol_function);
  options.put("Xtol.AbsoluteTolerance", optimizationOptions.atol_step);
  options.put("Xtol.RelativeTolerance", optimizationOptions.rtol_step);
  options.put("MaxEvaluations", optimizationOptions.maxEvals);
  options.put("Algorithm", optimizationOptions.algNLOPT);

  std::cout << "Algorithm: " << optimizationOptions.algNLOPT << std::endl;

  auto opt = std::make_shared<NLoptOptimizer>(globalCost, options);

  std::cout << "CREATED" << std::endl;

  // get the current coefficients---they make up the initial guess
  Eigen::VectorXd coefficients = cloud->GetCoefficients();
  std::cout << "GOT EM" << std::endl;
  std::cout << "cost: " << globalCost->Cost(coefficients) << std::endl;
  assert(opt);
  std::tie(coefficients, cost) = opt->Solve(std::vector<Eigen::VectorXd>(1, coefficients));

  std::cout << "DONE" << std::endl;

  // set the new coefficients
  cloud->SetCoefficients(coefficients);
  std::cout << "FINAL COST: " << cost << std::endl;

  return cost;*/

  /*assert(globalCost);

  // get the current coefficients---they make up the initial guess
  Eigen::VectorXd coefficients = cloud->GetCoefficients();

  //const double initCost = globalCost->Cost(coefficients);
  const double initCost = 0.0;
  double prevCost = initCost;

  double lambda = 1.0;
  for( std::size_t iter=0; iter<optimizationOptions.maxEvals; ++iter ) {
    std::cout << std::endl << std::endl;
    std::cout << "cost: " << prevCost << std::endl;
    // compute the cost function gradient
    //const Eigen::VectorXd grad = globalCost->Gradient(coefficients);
    const Eigen::VectorXd grad;

    std::cout << "gradient norm: " << grad.norm() << std::endl;

    // if the gradient is small, break
    if( grad.norm()<optimizationOptions.atol_grad ) { break; }

    Eigen::VectorXd stepDir;
    double newCost = std::numeric_limits<double>::infinity();
    double alpha;
    //Eigen::VectorXd newCoeff;
    while( newCost>prevCost ) {
      // compute the step direction
      stepDir = StepDirection(coefficients, grad, lambda, optimizationOptions.useGaussNewtonHessian);
      //newCoeff = coefficients - StepDirection(coefficients, grad, lambda, optimizationOptions.useGaussNewtonHessian);

      // compute the updated cost
      //newCost = globalCost->Cost(coefficients-stepDir);
      std::cout << "new: " << newCost << " prev: " << prevCost << std::endl;

      //lambda *= (alpha<0.5? 10.0 : 0.5);
      std::cout << "lambda: " << lambda << std::endl;

      std::tie(alpha, newCost) = LineSearch(coefficients, stepDir, prevCost);
      std::cout << "alpha: " << alpha << " new cost: " << newCost << std::endl;

      break;
    }
    assert(stepDir.size()==coefficients.size());
    //assert(newCoeff.size()==coefficients.size());
    //coefficients = newCoeff;
    coefficients -= alpha*stepDir;
    prevCost = newCost;

    std::cout << std::endl;*/

    /*// do the line search
    double alpha, newCost;
    std::tie(alpha, newCost) = LineSearch(coefficients, stepDir, prevCost);
    std::cout << "alpha: " << alpha << " new cost: " << newCost << std::endl;

    // if we are not using the Gauss-Newton Hessian and the stepsize is small, try taking a step with the Gauss-Newton Hessian
    if( !optimizationOptions.useGaussNewtonHessian && alpha<optimizationOptions.atol_step ) {
      // compute a new step direction
      //stepDir = StepDirection(coefficients, grad,  1.0/(1.0+iter), true);
      stepDir = StepDirection(coefficients, grad, lambda, true);

      // do the line search
      std::tie(alpha, newCost) = LineSearch(coefficients, stepDir, prevCost);
    }

    // make a step
    if( alpha<1.0-1.0e-10 ) { stepDir *= alpha; }
    const double stepsize = stepDir.norm();
    prevCost = newCost;
    coefficients -= stepDir;*/

    // check convergence
    /*if(
      prevCost<optimizationOptions.atol_function |
      prevCost/std::max(1.0e-10, initCost)<optimizationOptions.rtol_function
    ) { break; }*/
    /*if(
      prevCost<optimizationOptions.atol_function |
      prevCost/std::max(1.0e-10, initCost)<optimizationOptions.rtol_function |
      stepsize<optimizationOptions.atol_step |
      stepsize/std::max(1.0e-10, coefficients.norm())<optimizationOptions.rtol_step
    ) { break; }*/
  //}

  /*// set the new coefficients
  cloud->SetCoefficients(coefficients);
  cost =  prevCost;
  std::cout << "FINAL COST: " << cost << std::endl;

  return cost;*/
}

double LocalFunctions::ComputeIndependentSupportPoints() {
  cost = 0.0;
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) { cost += (*point)->MinimizeUncoupledCost(); }

  std::cout << "!!!!!COMPUTED COST: " << cost << std::endl;
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
    if( (*point)->Coupled() ) { return std::make_shared<GlobalCost>(cloud, pt); }
  }
  return nullptr;
}
