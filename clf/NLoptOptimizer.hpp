#ifndef NLOPTOPTIMIZER_HPP_
#define NLOPTOPTIMIZER_HPP_

#include <MUQ/Optimization/NLoptOptimizer.h>

#include "clf/Optimizer.hpp"

namespace clf {

/// Use <tt>NLopt</tt> to solve the optimization problem
/**
<B>Configuration Parameters:</B>
Requires all of the configuration parameters in clf::Optimizer and
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"GradientTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the gradient norm. |
"FunctionTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the cost function. |
"MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations. |
"Algorithm"   | <tt>std::string</tt> | <tt>"LBFGS"</tt> | The NLopt algorithm (see <a href="muq.mit.edu">MUQ</a> documentation for a list of options). |
*/
template<typename MatrixType>
class NLoptOptimizer : public Optimizer<MatrixType> {
public:
  /**
  @param[in] cost The cost function that we need to minimize
  */
  inline NLoptOptimizer(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  Optimizer<MatrixType>(cost, pt),
  algorithm(pt.get<std::string>("Algorithm", "LBFGS"))
  {}

  virtual ~NLoptOptimizer() = default;

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override {
    // set the options
    boost::property_tree::ptree options;
    options.put("Ftol.AbsoluteTolerance", this->funcTol);
    options.put("Ftol.RelativeTolerance", 0.0);
    options.put("Xtol.AbsoluteTolerance", 0.0);
    options.put("Xtol.RelativeTolerance", 0.0);
    options.put("MaxEvaluations", this->maxEvals);
    options.put("Algorithm", algorithm);

    // minimize the cost
    auto opt = std::make_shared<muq::Optimization::NLoptOptimizer>(this->cost, options);
    double costVal;
    std::tie(beta, costVal) = opt->Solve(std::vector<Eigen::VectorXd>(1, beta));

    // if converged, the cost function should be small
    if( costVal<this->funcTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_FUNCTION_SMALL, costVal); }

    // if the cost function is not small enough, but the gradient is small we still have converged
    const Eigen::VectorXd grad = this->cost->Gradient(0, beta, Eigen::VectorXd::Ones(1).eval());
    if( grad.norm()<this->gradTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_GRADIENT_SMALL, costVal); }

    if( std::isnan(costVal) | std::isinf(costVal) ) { std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED, costVal); }

    // something might have gone wrong ...
    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONTINUE_RUNNING, costVal);
  }

private:

  const std::string algorithm;
};

typedef NLoptOptimizer<Eigen::MatrixXd> DenseNLoptOptimizer;
typedef NLoptOptimizer<Eigen::SparseMatrix<double> > SparseNLoptOptimizer;

} // namespace clf

#endif
