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
"GradientTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the gradient norm. |
"FunctionTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the cost function. |
"ParameterTolerance"   | <tt>double</tt> | <tt>1.0e-10</tt> | The tolerance for the parameter norm. |
"MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations. |
"MaximumJacobianEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of Jacobian evaluations. |
"MaximumIterations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of iterations. |
"InitialDampling"   | <tt>double</tt> | <tt>0.0</tt> | The value of the damping parameter at the first iteration. |
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
  initialDamping(pt.get<double>("InitialDamping", 0.0))
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
      const std::pair<Optimization::Convergence, double> convergenceInfo = Iteration(iter, damping, beta, costVec);

      // check for convergence
      if( convergenceInfo.first>0 ) { return convergenceInfo; }

      // check max function evals
      if( numCostEvals>this->maxEvals ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_NUM_COST_EVALS, convergenceInfo.second); }

      // check max jacobian evals
      if( numJacEvals>maxJacEvals ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED_MAX_NUM_JACOBIAN_EVALS, convergenceInfo.second); }

      // update damping parameter and previous cost
      damping *= (convergenceInfo.second<prevCost? 0.75 : 2.0);
      prevCost = convergenceInfo.second;
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
  @param[in,out] iter In: The current iteration number, Out: The incremeneted iterationnumber
  @param[in,out] beta In: The current parameter value, Out: The updated parameter value
  @param[in,out] costVec In: The cost given the current parameter value, Out: The cost at the next iteration
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Iteration(std::size_t& iter, double const damping, Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    // increment the iteration
    ++iter;

    // evaluate the cost at the initial guess
    EvaluateCost(beta, costVec);
    const double costVal = costVec.dot(costVec);
    if( costVal<this->funcTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_FUNCTION_SMALL, costVal); }

    // compute the Jacobian matrix
    MatrixType jac;
    Jacobian(beta, jac);
    if( (jac.adjoint()*costVec).norm()<this->gradTol ) { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED_GRADIENT_SMALL, costVal); }

    // form J^T costVec
    Eigen::VectorXd JTcost = jac.transpose()*costVec;

    // compute the QR factorization of the Jacobian matrix
    jac = jac.transpose()*jac;
    if( damping>1.0e-14 ) { AddScaledIdentity(damping, jac); }

    // solve the linear system to compute the step direction and take a step 
    beta -= this->SolveLinearSystem(jac, JTcost);

    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONTINUE_RUNNING, costVal);
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

  /// The number of cost function evaluations
  std::size_t numCostEvals = 0;

  /// The number of Jacobian evaluations
  std::size_t numJacEvals = 0;

  /// The dampling parameter at the first iteration 
  const double initialDamping;
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
