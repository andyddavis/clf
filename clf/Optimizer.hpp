#ifndef OPTIMIZER_HPP_
#define OPTIMIZER_HPP_

#include <boost/property_tree/ptree.hpp>

#include "clf/CostFunction.hpp"

namespace clf {

namespace Optimization {
/// Information about whether or not the algorithm converged
enum Convergence {
  /// Hit the maximum number of Jacobian evaluations
  FAILED_MAX_NUM_JACOBIAN_EVALS = -3,

  /// Hit the maximum number of function evaluations
  FAILED_MAX_NUM_COST_EVALS = -2,

  /// Failed for some unknown (or unspecified reason)
  FAILED = -1,

  /// The algorithm is currently running
  CONTINUE_RUNNING = 0,

  /// Converged for some unknown (or unspecified reason)
  CONVERGED = 1,

  /// Converged because the cost function is small
  CONVERGED_FUNCTION_SMALL = 2,

  /// Converged because the gradient is small
  CONVERGED_GRADIENT_SMALL = 3,
};
} // namespace Optimization

/// A generic optimization algorithm
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"GradientTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the gradient norm. |
"FunctionTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the cost function. |
"MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations. |
*/
template<typename MatrixType>
class Optimizer {
public:

  /**
  @param[in] cost The cost function that we need to minimize
  @param[in] pt Options for the algorithm
  */
  inline Optimizer(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  cost(cost),
  gradTol(pt.get<double>("GradientTolerance", 1.0e-10)),
  funcTol(pt.get<double>("FunctionTolerance", 1.0e-10)),
  maxEvals(pt.get<std::size_t>("MaximumFunctionEvaluations", 1000))
  {}

  virtual ~Optimizer() = default;

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) = 0;

  /// The tolerance for convergence based on the gradient norm
  const double gradTol;

  /// The tolerance for convergence based on the function norm
  const double funcTol;

  /// The maximum number of cost function evaluations
  const std::size_t maxEvals;

protected:

  /// The cost function that we need to minimize
  std::shared_ptr<CostFunction<MatrixType> > cost;

private:
};

} // namespace clf

#endif
