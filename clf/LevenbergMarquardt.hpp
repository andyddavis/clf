#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include <boost/property_tree/ptree.hpp>

#include <Eigen/QR>

#include "clf/Optimizer.hpp"

namespace clf {

/// An implementation of the Levenberg Marquardt algorithm
/**
Used to minimize cost functions that are children of clf::CostFunction.

<B>Configuration Parameters:</B>
Requires all of the configuration parameters in clf::Optimizer and
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"GradientTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the gradient norm |
"FunctionTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the cost function |
"ParameterTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the parameter norm |
"MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations |
"MaximumJacobianEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of Jacobian evaluations |
"MaximumIterations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of iterations |
"InitialDampling"   | <tt>double</tt> | <tt>1.0</tt> | The value of the damping parameter at the first iteration |
"DampingShrinkFactor"   | <tt>double</tt> | <tt>0.5</tt> | The factor multiplying the damping factor if the error has decreased |
"DampingGrowFactor"   | <tt>double</tt> | <tt>1.5</tt> | The factor multiplying the damping factor if the error has increased |
"LineSearchFactor"   | <tt>double</tt> | <tt>0.5</tt> | Each iteration of the line search parameter decreases the step size by this factor |
"MaxLineSearchSteps"   | <tt>std::size_t</tt> | <tt>10</tt> | The maximum number of line search iterations |
*/
template<typename MatrixType>
class LevenbergMarquardt : public Optimizer<MatrixType> {
public:

  /**
  @param[in] cost The cost function that we need to minimize
  @param[in] pt Options for the algorithm
  */
  inline LevenbergMarquardt(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  Optimizer<MatrixType>(cost, pt),
  betaTol(pt.get<double>("ParameterTolerance", 1.0e-10)),
  maxJacEvals(pt.get<std::size_t>("MaximumJacobianEvaluations", 1000)),
  maxIters(pt.get<std::size_t>("MaximumIterations", 1000)),
  initialDamping(pt.get<double>("InitialDamping", 1.0)),
  dampingShrinkFactor(pt.get<double>("DampingShrinkFactor", 0.5)),
  dampingGrowFactor(pt.get<double>("DampingGrowFactor", 1.5)),
  lineSearchFactor(pt.get<double>("LineSearchFactor", 0.5)),
  maxLineSearchSteps(pt.get<std::size_t>("MaxLineSearchSteps", 10))
  {}

  virtual ~LevenbergMarquardt() = default;

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override {
    Eigen::VectorXd costVec;
    return Minimize(beta, costVec);
  }

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  @param[out] costVec The evaluation of the cost function at the optimizal point
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    assert(beta.size()==this->cost->inputDimension);

    // reset the algorithm parameters (e.g., function evaluation counters)
    ResetParameters();

    // evaluate the cost at the initial guess
    EvaluateCost(beta, costVec);
    double prevCost = costVec.dot(costVec);

    std::size_t iter = 0;
    double damping = initialDamping;
    while( iter<maxIters ) {
      const std::pair<Optimization::Convergence, double> convergenceInfo = Iteration(prevCost, damping, beta, costVec);

      // check for convergence
      if( convergenceInfo.first>0 ) { return convergenceInfo; }

      // check max function evals
      if( numCostEvals>this->maxEvals ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_NUM_COST_EVALS, convergenceInfo.second); }

      // check max jacobian evals
      if( numJacEvals>maxJacEvals ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_NUM_JACOBIAN_EVALS, convergenceInfo.second); }

      // update damping parameter and previous cost
      damping *= (convergenceInfo.second<prevCost? dampingShrinkFactor : dampingGrowFactor);
      prevCost = convergenceInfo.second;

      // increment the iteration
      ++iter;
    }
  }

  /// The tolerance for convergence based on the parameter norm
  const double betaTol;

  /// The maximum number of Jacobian evaluations
  const std::size_t maxJacEvals;

  /// The maximum number of iterations
  const std::size_t maxIters;

protected:

  /// Add an scaled identity to a matrix
  /**
  Sparse and dense matrices do this slightly differently
  @param[in] scale The scale that multiplies the identity matrix
  @param[in, out] mat The matrix we are added the scaled identity to
  */
  virtual void AddScaledIdentity(double const scale, MatrixType& mat) const = 0;

private:

  /// Reset the parameters for the Levenberg Marquardt algorithm
  /**
  Resets the count of the number of cost function and Jacobian evaluations.
  */
  inline void ResetParameters() {
    numCostEvals = 0;
    numJacEvals = 0;
  }

  /// Perform one iteration of the Levenberg Marquardt algorithm
  /**
  @param[in] costVal The current value of the cost function
  @param[in,out] beta In: The current parameter value, Out: The updated parameter value
  @param[in,out] costVec In: The cost given the current parameter value, Out: The cost at the next iteration
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Iteration(double const costVal, double const damping, Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    // evaluate the cost at the initial guess
    if( costVal<this->funcTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_FUNCTION_SMALL, costVal); }

    // compute the Jacobian matrix
    MatrixType jac;
    Jacobian(beta, jac);
    if( (jac.adjoint()*costVec).norm()<this->gradTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_GRADIENT_SMALL, costVal); }

    // compute the step direction
    const Eigen::VectorXd stepDir = StepDirection(damping, jac, costVec);

    // do a line search to make a step and return the result
    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONTINUE_RUNNING, LineSearch(costVal, stepDir, beta, costVec));
  }

  /// Evaluate the cost function
  /**
  @param[in] beta The parameter value
  @param[out] costVec The \f$i^{th}\f$ componenent is the evaluation of the function \f$f_i(\boldsymbol{\beta})\f$
  */
  inline void EvaluateCost(Eigen::VectorXd const& beta, Eigen::VectorXd& costVec) {
    costVec = this->cost->CostVector(beta);
    ++numCostEvals;
  }

  /// Compute the Jacobian matrix
  /**
  Also, increment the counter for the number of times we needed to compute the Jacobian matrix
  @param[in] beta The parameter value
  @param[out] jac The Jacobian matrix
  */
  inline void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) {
    this->cost->Jacobian(beta, jac);
    ++numJacEvals;
  }

  /// Compute the step direction
  /**
  @param[in] damping The damping constant that scales the step size
  @param[in, out] jac In: The Jacobian matrix \f$J\f$ (gradient of each penalty function), Out: The symmetric matrix \f$J^{\top} J\f$
  @param[in, out] costVec In: The vector \f$c\f$ of penalty function evaluations Out: The vector \f$J^{\top} c\f$
  \return The step direction
  */
  inline Eigen::VectorXd StepDirection(double const damping, MatrixType& jac, Eigen::VectorXd& costVec) const {
    // compute the QR factorization of the Jacobian matrix
    if( damping>1.0e-14 ) {
      costVec = jac.transpose()*costVec;
      jac = jac.transpose()*jac;
      AddScaledIdentity(damping, jac); 
    }

    auto linSolve = std::make_shared<LinearSolver<MatrixType> >(jac, this->linSolver, damping<=1.0e-14);
    return linSolve->Solve(costVec);
  }

  /// Do a line search
  /**
  @param[in] costVal The current value of the cost function
  @param[in] stepDir The step direction
  @param[in, out] beta In: The current parameter value, Out: the updated parameter value
  @param[out] costVec The penalty functions evaluated at the updated parameter value
  \return The new value of the cost function
  */
  inline double LineSearch(double const costVal, Eigen::VectorXd const& stepDir, Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    double alpha = 1.0;

    // evaluate the cost at the initial guess
    EvaluateCost(beta - stepDir, costVec);
    double newCost = costVec.dot(costVec);
    std::size_t iter = 0;
    while( newCost>costVal & iter++<maxLineSearchSteps ) {
      alpha *= lineSearchFactor;
      EvaluateCost(beta - alpha*stepDir, costVec);
      newCost = costVec.dot(costVec);
    }

    // take a step
    beta -= alpha*stepDir;

    return newCost;
  }

  /// The number of cost function evaluations
  std::size_t numCostEvals = 0;

  /// The number of Jacobian evaluations
  std::size_t numJacEvals = 0;

  /// The damping parameter at the first iteration
  const double initialDamping;

  /// The factor multiplying the damping factor if the error has decreased
  const double dampingShrinkFactor;

  /// The factor multiplying the damping factor if the error has increased
  const double dampingGrowFactor;

  /// Each iteration of the line search parameter decreases the step size by this factor
  const double lineSearchFactor;

  /// The maximum number of line search iterations
  const std::size_t maxLineSearchSteps;
};

/// An implementation of the Levenberg Marquardt algorithm using a dense Jacobian matrix
/**
See clf::LevenbergMarquardt for parameter options.
*/
class DenseLevenbergMarquardt : public LevenbergMarquardt<Eigen::MatrixXd> {
public:
  /**
  @param[in] cost The cost function that we need to minimize
  @param[in] pt Options for the algorithm
  */
  DenseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::MatrixXd> > const& cost, boost::property_tree::ptree const& pt);

  virtual ~DenseLevenbergMarquardt() = default;

protected:

  /// Add an scaled identity to a matrix
  /**
  @param[in] scale The scale that multiplies the identity matrix
  @param[in, out] mat The matrix we are added the scaled identity to
  */
  virtual void AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const override;

private:
};

/// An implementation of the Levenberg Marquardt algorithm using a sparse Jacobian matrix
/**
See clf::LevenbergMarquardt for parameter options.
*/
class SparseLevenbergMarquardt : public LevenbergMarquardt<Eigen::SparseMatrix<double> > {
public:

  /**
  @param[in] cost The cost function that we need to minimize
  @param[in] pt Options for the algorithm
  */
  SparseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::SparseMatrix<double> > > const& cost, boost::property_tree::ptree const& pt);

  virtual ~SparseLevenbergMarquardt() = default;

protected:

  /// Add an scaled identity to a matrix
  /**
  @param[in] scale The scale that multiplies the identity matrix
  @param[in, out] mat The matrix we are added the scaled identity to
  */
  virtual void AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const override;

private:
};

} // namespace clf

#endif
