#ifndef COSTFUNCTION_HPP_
#define COSTFUNCTION_HPP_

#include "clf/PenaltyFunction.hpp"

namespace clf {

/// A cost function that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt)
/**
   The Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) solves optimization problems of the form 
   \f{equation*}{
      \min_{\beta \in \mathbb{R}^{d}} \sum_{i=1}^{m} c_i(\beta) \cdot c_i(\beta),
   \f}
   where \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$ are penalty functions (see clf::PenaltyFunction).
*/
template<typename MatrixType>
class CostFunction {
public:

  /**
     @param[in] penaltyFuncs A vector of penalty functions such that the \f$i^{\text{th}}\f$ entry is \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
   */
  inline CostFunction(PenaltyFunctions<MatrixType> const& penaltyFunctions) :
    numPenaltyFunctions(penaltyFunctions.size()),
    numTerms(NumTerms(penaltyFunctions)),
    penaltyFunctions(penaltyFunctions)
  {
    // make sure all of the penalty functions have the same input dimension 
    assert(penaltyFunctions.size()>0);
    for( auto it=penaltyFunctions.begin()+1; it!=penaltyFunctions.end(); ++it ) { assert(penaltyFunctions[0]->indim==(*it)->indim); }
  }

  /**
     @param[in] penaltyFunction The penalty function \f$c_0: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
   */
  inline CostFunction(std::shared_ptr<PenaltyFunction<MatrixType> > const& penaltyFunction) :
    numPenaltyFunctions(1),
    numTerms(penaltyFunction->outdim),
    penaltyFunctions(PenaltyFunctions<MatrixType>(1, penaltyFunction))
  {}

  virtual ~CostFunction() = default;

  /// Return a vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
   */
  inline Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) const {
    assert(beta.size()==InputDimension());

    Eigen::VectorXd output(numTerms);
    std::size_t start = 0;
    for( const auto& it : penaltyFunctions ) {
      output.segment(start, it->outdim) = it->Evaluate(beta);
      start += it->outdim;
    }
    assert(start==numTerms);

    return output;
  }

  /// The number of parameters \f$d\f$ for the cost function
  /**
     \return The number of parameters \f$d\f$ for the cost function
   */
  inline std::size_t InputDimension() const { return penaltyFunctions[0]->indim; } 

  /// Compute the Jacobian of the penalty functions \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  /**
     The Jacobian of the penalty function is 
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     \nabla_{\beta} c_0(\beta) \\
     \vdots \\
     \nabla_{\beta} c_m(\beta) 
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
   */
  virtual MatrixType Jacobian(Eigen::VectorXd const& beta) const = 0;

  /// Compute the gradient of the cost function \f$2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)\f$ given that we have already computed the Jacobian (CostFunction::Jacobian) and the cost (CostFunction::Evaluate);
  /**
     The gradient of the cost function is \f$2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)\f$, where \f$c_{i,j}\f$ is the \f$j^{\text{th}}\f$ component of the \f$i^{\text{th}}\f$ penalty function.
     @param[in] cost The penalty function evaluations (output of CostFunction::Evaluate)
     @param[in] jac The Jacobian (output of CostFunction::Jacobian)
     \return The gradient of the cost function 
   */
  inline Eigen::VectorXd Gradient(Eigen::VectorXd const& cost, MatrixType const& jac) const { return 2.0*jac.adjoint()*cost; }

  /// Compute the gradient of the cost function \f$2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)\f$
  /**
     The gradient of the cost function is \f$2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)\f$, where \f$c_{i,j}\f$ is the \f$j^{\text{th}}\f$ component of the \f$i^{\text{th}}\f$ penalty function.
     @param[in] beta The input parameters \f$\beta\f$
     \return The gradient of the cost function 
   */
  inline Eigen::VectorXd Gradient(Eigen::VectorXd const& beta) const { return Gradient(Evaluate(beta), Jacobian(beta)); }

  /// Compute the Hessian of the cost function 
  /**
     The Hessian of the penalty function is 
     \f{equation*}{
     H = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \left( \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top} + c_{i,j}(\beta) \nabla_{\beta}^2 c_{i,j}(\beta) \right),
     \f}
     where \f$c_{i,j}(\beta)\f$ is the \f$j^{\text{th}}\f$ component of the \f$i^{\text{th}}\f$ penalty function. This function either computes the Hessian or the Gauss-Newton approximation to the Hessian:
     \f{equation*}{
     H_{\text{gn}} = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] gn True: compute the Gauss-Newton Hessian, False (default): compute the true Hessian
     \return The Hessian (or Gauss-Newton Hessian) of the cost function
   */
  inline MatrixType Hessian(Eigen::VectorXd const& beta, bool const gn = false) const { return Hessian(beta, Jacobian(beta), gn); } 

  /// Compute the Hessian of the cost function given that we know the Jacobian
  /**
     The Hessian of the penalty function is 
     \f{equation*}{
     H = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \left( \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top} + c_{i,j}(\beta) \nabla_{\beta}^2 c_{i,j}(\beta) \right),
     \f}
     where \f$c_{i,j}(\beta)\f$ is the \f$j^{\text{th}}\f$ component of the \f$i^{\text{th}}\f$ penalty function. This function either computes the Hessian or the Gauss-Newton approximation to the Hessian:
     \f{equation*}{
     H_{\text{gn}} = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] jac The Jacobian (output of CostFunction::Jacobian)
     @param[in] gn True: compute the Gauss-Newton Hessian, False (default): compute the true Hessian
     \return The Hessian (or Gauss-Newton Hessian) of the cost function
   */
  inline MatrixType Hessian(Eigen::VectorXd const& beta, MatrixType const& jac, bool const gn = false) const {
    if( gn ) { return 2.0*jac.adjoint()*jac; }
    return Hessian(beta, Evaluate(beta), jac, gn);
  }

  /// Compute the Hessian of the cost function given that we know the Jacobian and the penalty function evaluations
  /**
     The Hessian of the penalty function is 
     \f{equation*}{
     H = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \left( \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top} + c_{i,j}(\beta) \nabla_{\beta}^2 c_{i,j}(\beta) \right),
     \f}
     where \f$c_{i,j}(\beta)\f$ is the \f$j^{\text{th}}\f$ component of the \f$i^{\text{th}}\f$ penalty function. This function either computes the Hessian or the Gauss-Newton approximation to the Hessian:
     \f{equation*}{
     H_{\text{gn}} = 2 \sum_{i=1}^{m} \sum_{j=1}^{n_i} \nabla_{\beta} c_{i,j}(\beta) \nabla_{\beta} c_{i,j}(\beta)^{\top}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] cost The penalty function evaluations (output of CostFunction::Evaluate)
     @param[in] jac The Jacobian (output of CostFunction::Jacobian)
     @param[in] gn True: compute the Gauss-Newton Hessian, False (default): compute the true Hessian
     \return The Hessian (or Gauss-Newton Hessian) of the cost function
   */
  inline MatrixType Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& cost, MatrixType const& jac, bool const gn = false) const {
    MatrixType hess = jac.adjoint()*jac;
    if( gn ) { return 2.0*hess; }

    std::size_t start = 0;
    for( const auto& it : penaltyFunctions ) { 
      hess += it->Hessian(beta, cost.segment(start, it->outdim)); 
      start += it->outdim;
    }

    return 2.0*hess;
  }

  /// The number of terms in the sum \f$\bar{n}\f$
  const std::size_t numTerms;

  /// The number of penalty functions 
  /**
     This is the number of functions \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$ 
   */
  const std::size_t numPenaltyFunctions;

protected:

  /// The penalty functions that define the cost function 
  const PenaltyFunctions<MatrixType> penaltyFunctions;

private:

  /// Compute the number of terms in the sum \f$\bar{n}\f$
  /**
     Each penalty function can have multiple outputs. This is the total number of terms in the cost function \f$\bar{n} = \sum_{i=0}^{m} n_i\f$.
     \return The number of terms in the sum \f$\bar{n}\f$
   */
  static inline std::size_t NumTerms(PenaltyFunctions<MatrixType> const& penaltyFunctions) { 
    std::size_t num = 0; 
    for( const auto& it : penaltyFunctions ) { num += it->outdim; }
    return num;
  } 
};

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
class DenseCostFunction : public CostFunction<Eigen::MatrixXd> {
public:
  
  /**
     @param[in] penaltyFuncs A vector of penalty functions such that the \f$i^{\text{th}}\f$ entry is \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  DenseCostFunction(DensePenaltyFunctions const& penaltyFunctions);

  /**
     @param[in] penaltyFunction The penalty function \f$c_0: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  DenseCostFunction(std::shared_ptr<DensePenaltyFunction> const& penaltyFunction);

  virtual ~DenseCostFunction() = default; 

  /// Compute the Jacobian of the penalty functions \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  /**
     The Jacobian of the penalty function is 
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     \nabla_{\beta} c_0(\beta) \\
     \vdots \\
     \nabla_{\beta} c_m(\beta) 
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  */
  virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) const final override;

private:
};

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
class SparseCostFunction : public CostFunction<Eigen::SparseMatrix<double> > {
public:
  
  /**
     @param[in] penaltyFuncs A vector of penalty functions such that the \f$i^{\text{th}}\f$ entry is \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  SparseCostFunction(SparsePenaltyFunctions const& penaltyFunctions);

  /**
     @param[in] penaltyFunction The penalty function \f$c_0: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  SparseCostFunction(std::shared_ptr<SparsePenaltyFunction> const& penaltyFunction);

  virtual ~SparseCostFunction() = default; 

  /// Compute the Jacobian of the penalty functions \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  /**
     The Jacobian of the penalty function is 
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     \nabla_{\beta} c_0(\beta) \\
     \vdots \\
     \nabla_{\beta} c_m(\beta) 
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  */
  virtual Eigen::SparseMatrix<double> Jacobian(Eigen::VectorXd const& beta) const final override;

private:
};

} // namespace clf 

#endif
