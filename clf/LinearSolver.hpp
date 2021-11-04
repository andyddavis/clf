#ifndef LINEARSOLVER_HPP_
#define LINEARSOLVER_HPP_

#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>

namespace clf {

/// Which solver should we use to solve the system?
enum LinearSolverType {
  /// Use a Cholesky solver 
  Cholesky, 
  
  /// Use a Cholesky solver with pivoting
  CholeskyPivot, 
  
  /// Use a QR solver
  QR,
  
  /// Use an LU solver
  LU
};

/// Solve a linear system of the form \f$A x = b\f$. If \f$A\f$ is not square solve the least-squares problem \f$A^{\top} A x = A^{\top} b\f$.
template<typename MatrixType>
class LinearSolver {
public:

  /// The type for the Cholesky solver
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::LLT<Eigen::MatrixXd>, Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > >::type SolverCholesky;

  /// The type for the Cholesky solver with pivoting
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::LDLT<Eigen::MatrixXd>, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > >::type SolverCholeskyPivot;

  /// The type for the LU solver
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::PartialPivLU<Eigen::MatrixXd>, Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > >::type SolverLU;

  /// The type for the QR solver
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>, Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > >::type SolverQR;

  /// Compute the matrix decomposition at construction
  /**
  @param[in] A The matrix that defines the linear system
  @param[in] solver The type of solver we want to use (default to Cholesky)
  @param[in] leastSq <tt>true</tt> (default): Solve the least squares problem \f$A^{\top} A x = A^{\top} b\f$; <tt>false</tt>: Solve the linear system \f$A x = b\f$ directly (in the QR case \f$A\f$ may not be square and this becomes a least squares problem still).
  */
  inline LinearSolver(MatrixType const& A, LinearSolverType const solver = Cholesky, bool const leastSq = true) :
    solver(solver),
    leastSq(leastSq)
  {
    switch( solver ) {
    case LinearSolverType::CholeskyPivot:
      if( !leastSq ) { assert(A.rows()==A.cols()); }
      solverCholeskyPivot.emplace((leastSq? A.transpose()*A : A));
      if( leastSq ) { matrix = A; }
      break;
    case LinearSolverType::QR:
      solverQR.emplace(A);
      break;      
    case LinearSolverType::LU:
      if( !leastSq ) { assert(A.rows()==A.cols()); }
      solverLU.emplace((leastSq? A.transpose()*A : A));
      if( leastSq ) { matrix = A; }
      break;
    default:
      if( !leastSq ) { assert(A.rows()==A.cols()); }
      solverCholesky.emplace((leastSq? A.transpose()*A : A));
      if( leastSq ) { matrix = A; }
    }
  }

  virtual ~LinearSolver() = default;

  /// Solve the linear system given the right hand side \f$b\f$
  /**
  @param[in] b The right hand side \f$b\f$
  \return The solution \f$x\f$
  */
  inline Eigen::VectorXd Solve(Eigen::VectorXd const& b) {
    switch( solver ) {
    case LinearSolverType::CholeskyPivot:
      return solverCholeskyPivot->solve((leastSq? matrix->transpose()*b : b));
    case LinearSolverType::QR:
      return SolveQR(b);
    case LinearSolverType::LU:
      return solverLU->solve((leastSq? matrix->transpose()*b : b));
    default:
      return solverCholesky->solve((leastSq? matrix->transpose()*b : b));
    }
  }

  // Solve a linear system \f$A x = b\f$ using a \f$QR\f$ decomposition
  /**
  If \f$A\f$ is not full rank, return the least-squares solution.
  @param[in] rhs The right hand side \f$b\f$
  \return The solution \f$x\f$
  */
  inline Eigen::VectorXd SolveQR(Eigen::VectorXd const& rhs) const {
    const Eigen::Index rank = solverQR->rank();
    std::cout << "RANK: " << rank << std::endl;

    Eigen::VectorXd soln(solverQR->cols());
    soln.tail(soln.size()-rank).setZero();
    soln.head(rank) = solverQR->matrixR().topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solve((solverQR->matrixQ().adjoint()*rhs).head(rank));
    soln = solverQR->colsPermutation()*soln;

    return soln;
  }

  /// The type of solver we are using
  const LinearSolverType solver;

  /// <tt>true</tt> (default): Solve the least squares problem \f$A^{\top} A x = A^{\top} b\f$; <tt>false</tt>: Solve the linear system \f$A x = b\f$ directly (in the QR case \f$A\f$ may not be square and this becomes a least squares problem still).
  const bool leastSq;

private:

  /// The matrix \f$A\f$ that defines the linear system 
  std::optional<MatrixType> matrix;

  /// The Cholesky solver used to compute the solution 
  /**
  The decomposition is precomputed at construction if we are using a Cholesky solve.
  */
  std::optional<SolverCholesky> solverCholesky;

  /// The Cholesky solver with pivoting used to compute the solution 
  /**
  The decomposition is precomputed at construction if we are using a Cholesky solve with pivoting.
  */
  std::optional<SolverCholeskyPivot> solverCholeskyPivot;

  /// The QR solver used to compute the solution 
  /**
  The decomposition is precomputed at construction if we are using a QR solver.
  */
  std::optional<SolverQR> solverQR;

  /// The LU solver used to compute the solution 
  /**
  The decomposition is precomputed at construction if we are using a LU solver.
  */
  std::optional<SolverLU> solverLU;

};

typedef LinearSolver<Eigen::MatrixXd> DenseLinearSolver;
typedef LinearSolver<Eigen::SparseMatrix<double> > SparseLinearSolver;

} // namespace clf 

#endif
