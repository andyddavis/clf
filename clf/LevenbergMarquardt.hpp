#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include "clf/Parameters.hpp"
#include "clf/CostFunction.hpp"

namespace clf {

namespace Optimization {

/// Information about whether or not the algorithm converged
enum Convergence {
  /// Failed for some unknown or unspecified reason
  FAILED = -1,

  /// The algorithm is currently running
  CONTINUE_RUNNING = 0,
};

} // namespace Optimization

/// An implementation of the Levenberg Marquardt algorithm 
/**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations. |
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
  inline std::size_t NumParameters() const { return cost->indim; }

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

    //const std::size_t maxiter = para->Get<std::size_t>("MaximumFunctionEvaluations");

    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED, std::numeric_limits<double>::quiet_NaN());
  }
  
private:

  /// Reset the counters for the number of function/Jacobian/Hessian calls to zero
  inline void ResetCounters() {
    numCostEvals = 0;
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
};

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using dense matrices
typedef LevenbergMarquardt<Eigen::MatrixXd> DenseLevenbergMarquardt;

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using sparse matrices
typedef LevenbergMarquardt<Eigen::SparseMatrix<double> > SparseLevenbergMarquardt;

} // namespace clf

#endif
