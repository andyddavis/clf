#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include "clf/CostFunction.hpp"
#include "clf/LinearSolver.hpp"

namespace clf {

namespace Optimization {

/// Information about whether or not the algorithm converged
enum Convergence {
  /// Failed because the algorithm hit the maximum number of cost function Hessian evaluations
  FAILED_MAX_HESSIAN_EVALUATIONS = -5,

  /// Failed because the algorithm hit the maximum number of cost function Jacobian evaluations
  FAILED_MAX_JACOBIAN_EVALUATIONS = -4,

  /// Failed because the algorithm hit the maximum number of cost function evaluations
  FAILED_MAX_COST_EVALUATIONS = -3,

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

  /// Converged because the cost function gradient is below a specified value
  CONVERGED_GRADIENT_SMALL = 3,
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
   "MaximumIterations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of interations (see clf::LevenbergMarquardt::maxIterations_DEFAULT). |
   "MaximumCostFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>10000</tt> | The maximum number of cost function evaluations (see clf::LevenbergMarquardt::maxCostEvals_DEFAULT). |
   "MaximumJacobianEvaluations"   | <tt>std::size_t</tt> | <tt>10000</tt> | The maximum number of cost function Jacobian evaluations (see clf::LevenbergMarquardt::maxJacEvals_DEFAULT). |
   "MaximumHessianEvaluations"   | <tt>std::size_t</tt> | <tt>10000</tt> | The maximum number of cost function Hessian evaluations (see clf::LevenbergMarquardt::maxHessEvals_DEFAULT). |
   "FunctionTolerance"   | <tt>double</tt> | <tt>0.0</tt> | The tolerance for the cost function value (see clf::LevenbergMarquardt::functionTolerance_DEFAULT). |
   "GradientTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the cost function gradient value (see clf::LevenbergMarquardt::gradientTolerance_DEFAULT). |
   "GaussNewtonHessian"   | <tt>bool</tt> | <tt>false</tt> | Should we use the Gauss Newton approximation of the Hessian? |
   "InitialDamping"   | <tt>double</tt> | <tt>1.0</tt> | The value of the damping parameter at the first iteration (see clf::LevenbergMarquardt::initialDamping_DEFUALT)) |
   "DampingShrinkFactor"   | <tt>double</tt> | <tt>0.1</tt> | The factor that the damping parameter shrinks if the cost is less than the previous iteration (see clf::LevenbergMarquardt::dampingShrinkFactor_DEFUALT)) |
   "DampingGrowFactor"   | <tt>double</tt> | <tt>2.0</tt> | The factor that the damping parameter grows if the cost is greater than the previous iteration (see clf::LevenbergMarquardt::dampingGrowFactor_DEFUALT)) |
   "LinearSolver"   | <tt>clf::LinearSolverType</tt> | <tt>clf::LinearSolverType::Cholesky</tt> | The linear solver used to solve linear systems (see clf::LevenbergMarquardt::linearSolver_DEFUALT)) |
   "LeastSquareSolve"   | <tt>bool</tt> | <tt>false</tt> | Use a least square solve to compute the step direction. |
   "MaximumLineSearchIterations"   | <tt>std::size_t</tt> | <tt>10</tt> | The maximum number of line search interations evaluations (see clf::LevenbergMarquardt::maxLineSearchIterations_DEFAULT). |
   "LineSearchFactor"   | <tt>double</tt> | <tt>0.5</tt> | The factor that the line search parameter shrink with each line search iteration (see clf::LevenbergMarquardt::lineSearchFactor_DEFAULT). |
 */
template<typename MatrixType>
class LevenbergMarquardt {
public:

  /**
     @param[in] cost The cost function that we are trying to minimize
     @param[in] para The parameters for this algorithm
   */
  inline LevenbergMarquardt(std::shared_ptr<const CostFunction<MatrixType> > const& cost, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>()) :
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
     Minimize without passing by reference for the sake of the python interface
     @param[in] beta The initial guess for the Levenberg Marquardt algorithm
     \return First: Information about convergence or failure Second: The current cost, Third: The parameter values that minimizes the cost function Fouth: The evaluation of the cost function at the optimal point
  */
  inline std::tuple<Optimization::Convergence, double, Eigen::VectorXd, Eigen::VectorXd> Minimize(Eigen::VectorXd const& beta) {
    std::tuple<Optimization::Convergence, double, Eigen::VectorXd, Eigen::VectorXd> result;
    std::get<2>(result) = beta;
    
    std::tie(std::get<0>(result), std::get<1>(result)) = Minimize(std::get<2>(result), std::get<3>(result));

    return result;
  }

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
     @param[in, out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The parameter values that minimizes the cost function
     @param[out] costVec The evaluation of the cost function at the optimal point
     \return First: Information about convergence or failure Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    assert(beta.size()==NumParameters());

    // reset the counters to zero 
    ResetCounters();

    // evaluate the cost at the intial guess 
    double prevCost = Evaluate(beta, costVec);

    const std::size_t maxiter = para->Get<std::size_t>("MaximumIterations", maxIterations_DEFAULT);
    std::size_t iter = 0;
    double damping = para->Get<double>("InitialDamping", initialDamping_DEFAULT);
    while( iter++<maxiter ) {
      // if the cost is sufficiently small, we have converged 
      if( prevCost<para->Get<double>("FunctionTolerance", functionTolerance_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_FUNCTION_SMALL, prevCost); }

      // take a single iteration 
      const std::pair<Optimization::Convergence, double> status = Iteration(beta, damping, prevCost, costVec);

      // if we have converged, return
      if( status.first>0 ) { return status; }

      // make sure we are not out of function evaluations
      if( numCostEvals>para->Get<std::size_t>("MaximumCostFunctionEvaluations", maxCostEvals_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_COST_EVALUATIONS, status.second); }
      if( numJacEvals>para->Get<std::size_t>("MaximumJacobianEvaluations", maxJacEvals_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_JACOBIAN_EVALUATIONS, status.second); }
      if( numHessEvals>para->Get<std::size_t>("MaximumHessianEvaluations", maxHessEvals_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_HESSIAN_EVALUATIONS, status.second); }

      // update the damping parameter
      damping *= (status.second<prevCost? para->Get<double>("DampingShrinkFactor", dampingShrinkFactor_DEFAULT) : para->Get<double>("DampingGrowFactor", dampingGrowFactor_DEFAULT));

      // update the previous cost
      prevCost = status.second;
    }

    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_ITERATIONS, std::numeric_limits<double>::quiet_NaN());
  }

protected:

  /// Add a scaled identity to a matrix
  /**
     Dense and sparse matrices do this slightly differently 
     @param[in] scale Add this number times and identity to the matrix 
     @param[in, out] mat We are adding to this matrix
   */
  virtual void AddScaledIdentity(double const scale, MatrixType& mat) const = 0;
  
private:

  /// Reset the counters for the number of function/Jacobian/Hessian calls to zero
  inline void ResetCounters() {
    numCostEvals = 0;
    numJacEvals = 0;
    numHessEvals = 0;
  }

  /// Evaluate the cost function, compute a vector of penalty functions
  /**
     @param[in] beta The The parameter vector
     @param[out] costVec The vector of each penalty function evaluation
     \return The cost function evaluation (the sum of the squared penalty functions)
  */
  inline double Evaluate(Eigen::VectorXd const& beta, Eigen::VectorXd& costVec) {
    // evalute the cost function 
    costVec = cost->Evaluate(beta);

    // increament the number of function evaluations 
    ++numCostEvals;

    // the cost function is the sum of the squared penalty terms
    return costVec.dot(costVec);
  }

  /// Evaluate the cost function Jacobian
  /**
     @param[in] beta The The parameter vector
     @param[out] jac The cost function Jacobian
  */
  inline void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) {
    assert(cost);

    // evalute the cost function Jacobian
    jac = cost->Jacobian(beta);

    // increament the number of Jacobian evaluations 
    ++numJacEvals;
  }

  /// Evaluate the cost function Hessian given that we already know the penalty function evaluations and the jacobian
  /**
     @param[in] beta The The parameter vector
     @param[in] costVec The cost function jacobian
     @param[in] jac The cost function Jacobian
     @param[out] hess The cost function Hessian
  */
  inline void Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& costVec, MatrixType const& jac, MatrixType& hess) {
    assert(cost);

    // evalute the cost function Hessian
    hess = cost->Hessian(beta, costVec, jac, para->Get<bool>("GaussNewtonHessian", false));

    // increament the number of Jacobian evaluations 
    ++numHessEvals;
  }

  /// Perform one iteration of the Levenberg Marquardt algorithm
  /**
     @param[in,out] beta The parameter vector, in: guess before the iteration, out: updated guess after one iteration
     @param[in] damping The damping parameter for this iteration 
     @param[in] costVal The cost function evaluation 
     @param[in, out] costVec The penalty function evaluations at beta
     \return First: Information about convergence or failure, Second: The current cost
   */
  inline std::pair<Optimization::Convergence, double> Iteration(Eigen::VectorXd& beta, double const damping, double costVal, Eigen::VectorXd& costVec) {
    // compute the jacobian matrix
    MatrixType jac;
    Jacobian(beta, jac);

    // use the jacobian to compute the gardient of the cost function and check for gradient convergence
    const Eigen::VectorXd grad = cost->Gradient(costVec, jac);
    if( grad.norm()<para->Get<double>("GradientTolerance", gradientTolerance_DEFAULT) ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_GRADIENT_SMALL, costVal); }

    // compute the step direction
    const Eigen::VectorXd stepDir = StepDirection(beta, damping, costVec, grad, jac);

    // do a line search
    costVal = LineSearch(beta, stepDir, costVal, costVec);

    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONTINUE_RUNNING, costVal);
  }

  /// Do the line search and update the parameter values
  /**
     @param[in,out] beta The parameter vector, in: guess before the iteration, out: updated guess after one iteration
     @param[in] stepDir The step direction 
     @param[in] costVal The cost function evaluation 
     @param[in, out] costVec The penalty function evaluations at beta
     \return The new cost function evaluation
    */
  inline double LineSearch(Eigen::VectorXd& beta, Eigen::VectorXd const& stepDir, double const costVal, Eigen::VectorXd& costVec) {
    double alpha = 1.0;

    // evalute the cost function
    Eigen::VectorXd b = beta - alpha*stepDir;
    double newCost = Evaluate(b, costVec);

    const std::size_t maxiter = para->Get<std::size_t>("MaximumLineSearchIterations", maxLineSearchIterations_DEFAULT);
    std::size_t iter = 0;
    const double scale = para->Get<double>("LineSearchFactor", lineSearchFactor_DEFAULT);
    while( newCost>costVal & iter++<maxiter ) {
      alpha *= scale;
      b = beta - alpha*stepDir;
      newCost = Evaluate(b, costVec);
    }

    beta = b;

    return newCost;
  }

  /// Compute the step direction
  /**
     @param[in] beta The parameter vector
     @param[in] damping The damping parameter for this iteration 
     @param[in] costVec The penalty function evaluations at beta
     @param[in] grad The cost function gradient 
     @param[in] jac The cost function Jacobian
     \return The step direction
    */
  inline Eigen::VectorXd StepDirection(Eigen::VectorXd const& beta, double const damping, Eigen::VectorXd const& costVec, Eigen::VectorXd const& grad, MatrixType const& jac) {
    // compute the hessian
    MatrixType hess;
    Hessian(beta, costVec, jac, hess);

    // add the damping parameter
    AddScaledIdentity(damping, hess);

    // compute the step direction
    auto linSolve = std::make_unique<LinearSolver<MatrixType> >(hess, para->Get<clf::LinearSolverType>("LinearSolver", clf::LinearSolverType::Cholesky), para->Get<bool>("LeastSquareSolve", false));
    return linSolve->Solve(grad);
  }

  /// The number of function evaluations 
  /**
     Starts at zero and reset to zero each time LevenbergMarquardt::Minimize is called. After LevenbergMarquardt::Minimize is called, this is the number of times CostFunction::Evaluate was called.
   */
  std::size_t numCostEvals = 0; 

  /// The number of Jacobian evaluations 
  /**
     Starts at zero and reset to zero each time LevenbergMarquardt::Minimize is called. After LevenbergMarquardt::Minimize is called, this is the number of times CostFunction::Jacobian was called.
   */
  std::size_t numJacEvals = 0; 

  /// The number of Hessian evaluations 
  /**
     Starts at zero and reset to zero each time LevenbergMarquardt::Minimize is called. After LevenbergMarquardt::Minimize is called, this is the number of times CostFunction::Hessian was called.
   */
  std::size_t numHessEvals = 0; 

  /// The parameters for this algorithm
  std::shared_ptr<const Parameters> para;

  /// The cost function we are trying to minimize
  std::shared_ptr<const CostFunction<MatrixType> > cost;

  /// The default maximum number of iterations 
  inline static std::size_t maxIterations_DEFAULT = 1000;

  /// The default maximum number of cost function evaluations
  inline static std::size_t maxCostEvals_DEFAULT = 10000;

    /// The default maximum number of cost function Jacobian evaluations
  inline static std::size_t maxJacEvals_DEFAULT = 10000;

  /// The default maximum number of cost function Hessian evaluations
  inline static std::size_t maxHessEvals_DEFAULT = 10000;

  /// The default tolerance for the cost function tolerance
  inline static double functionTolerance_DEFAULT = 0.0;

  /// The default tolerance for the cost function gradient tolerance
  inline static double gradientTolerance_DEFAULT = 1.0e-10;

  /// The default value of the damping parameter at the first iteration
  inline static double initialDamping_DEFAULT = 1.0;

  /// The default factor that the damping parameter shrinks if the cost is less than the previous iteration
  inline static double dampingShrinkFactor_DEFAULT = 0.1;

  /// The default factor that the damping parameter grows if the cost is greater than the previous iteration
  inline static double dampingGrowFactor_DEFAULT = 2.0;

  /// The default linear solver used to solve linear systems 
  inline static clf::LinearSolverType linearSolver_DEFAULT = clf::LinearSolverType::Cholesky;

  /// The default maximum number of iterations 
  inline static std::size_t maxLineSearchIterations_DEFAULT = 10;

  /// The factor that the line search parameter shrink with each line search iteration
  inline static double lineSearchFactor_DEFAULT = 0.5;
};

} // namespace clf

#endif
