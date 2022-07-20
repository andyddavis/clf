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
     @param[out] jac The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
   */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

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
     @param[out] jac The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const final override;

private:
};

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
class SparseCostFunction : public CostFunction<Eigen::SparseMatrix<double> > {
public:
  
  /**
     @param[in] penaltyFuncs A vector of penalty functions such that the \f$i^{\text{th}}\f$ entry is \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  SparseCostFunction(SparsePenaltyFunctions const& penaltyFunctions);

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
     @param[out] jac The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const final override;

private:
};

} // namespace clf 

#endif
