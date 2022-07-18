#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include "clf/CostFunction.hpp"

namespace clf {

namespace Optimization {

/// Information about whether or not the algorithm converged
enum Convergence {
  /// Failed because the algorithm hit the maximum number of iterations
  FAILED_MAX_ITERATIONS = -2,

  /// Failed for some unknown or unspecified reason
  FAILED = -1,

  /// The algorithm is currently running
  CONTINUE_RUNNING = 0,

  /// Converged for some unknown or unspecified reason
  CONVERGED = 1,

  /// Converged because the cost function is below a specified value
  CONVERGED_FUNCTION_SMALL = 2,

};

} // namespace Optimization

/// An implementation of the Levenberg Marquardt algorithm 
/**
   Solve an optimization problem with the form 
   \f{equation*}{
      \min_{\beta \in \mathbb{R}^{d}} \sum_{i=1}^{m} c_i(\beta)^2
   \f}
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "MaximumNumIterations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of interations evaluations (see clf::LevenbergMarquardt::maxIterations_DEFAULT). |
   "FunctionTolerance"   | <tt>double</tt> | <tt>0.0</tt> | The tolerance for the cost function value (see clf::LevenbergMarquardt::functionTolerance_DEFAULT). |
 */
template<typename MatrixType>
class LevenbergMarquardt {
public:

  /**
     @param[in] cost The cost function that we are trying to minimize
     @param[in] para The parameters for this algorithm
   */
  inline LevenbergMarquardt(std::shared_ptr<const CostFunction<MatrixType> > const& cost, std::shared_ptr<Parameters> const& para) :
    para(para),
    cost(cost)
  {}

  virtual ~LevenbergMarquardt() = default;

  /// The number of parameters for the cost function 
  /**
     \return The number of parameters for the cost function
   */
  inline std::size_t NumParameters() const { return cost->InputDimension(); }

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
     @param[in, out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The
     paramter values that minimizes the cost function
     @param[out] costVec The evaluation of the cost function at the optimal point
     \return First: Information about convergence or failure , Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    assert(beta.size()==NumParameters());

    // reset the counters to zero 
    ResetCounters();

    // evaluate the cost at the intial guess 
    double prevCost = EvaluateCost(beta, costVec);

    const std::size_t maxiter = para->Get<std::size_t>("MaximumNumIterations", maxIterations_DEFAULT);
    std::size_t iter = 0;
    while( iter++<maxiter ) {
      // if the cost is sufficiently small, we have converged 
      if( prevCost<para->Get<double>("FunctionTolerance", functionTolerance_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_FUNCTION_SMALL, prevCost); }

      Iteration();

    }

    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_ITERATIONS, std::numeric_limits<double>::quiet_NaN());
  }
  
private:

  /// Reset the counters for the number of function/Jacobian/Hessian calls to zero
  inline void ResetCounters() {
    numCostEvals = 0;
  }

  /// Evaluate the cost function, compute a vector of penalty functions
  /**
     @param[in] beta The The parameter vector
     @param[out] costVec The vector of each penalty function evaluation
     \return The cost function evaluation (the sum of the squared penalty functions)
  */
  inline double EvaluateCost(Eigen::VectorXd const& beta, Eigen::VectorXd& costVec) {
    // evalute the cost function 
    costVec = cost->Evaluate(beta);

    // increatement the number of function evaluations 
    ++numCostEvals;

    // the cost function is the sum of the squared penalty terms
    return costVec.dot(costVec);
  }

  /// Perform one iteration of the Levenberg Marquardt algorithm
  inline void Iteration() const {
    // compute the jacobian matrix
  }

  /// The number of function evaluations 
  /**
     Starts at zero and reset to zero each time LevenbergMarquardt::Minimize is called. After LevenbergMarquardt::Minimize is called, this is the number of times CostFunction::Evaluate was called.
   */
  std::size_t numCostEvals = 0; 

  /// The parameters for this algorithm
  std::shared_ptr<const Parameters> para;

  /// The cost function we are trying to minimize
  std::shared_ptr<const CostFunction<MatrixType> > cost;

  /// The default maximum number of iterations 
  inline static std::size_t maxIterations_DEFAULT = 1000;

  /// The default tolerance for the cost function tolerance
  inline static double functionTolerance_DEFAULT = 0.0;

};

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using dense matrices
typedef LevenbergMarquardt<Eigen::MatrixXd> DenseLevenbergMarquardt;

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using sparse matrices
typedef LevenbergMarquardt<Eigen::SparseMatrix<double> > SparseLevenbergMarquardt;

} // namespace clf

#endif
