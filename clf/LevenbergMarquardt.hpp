#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include <boost/property_tree/ptree.hpp>

#include <Eigen/QR>

#include "clf/CostFunction.hpp"

namespace clf {

/// An implementation of the Levenberg Marquardt algorithm
/**
Used to minimize cost functions that are children of clf::CostFunction. This implementation of the Levenberg Marquardt algorithm is only a slight modification from the implementation in the Eigen-Unsupported code base (https://eigen.tuxfamily.org/dox/unsupported/classEigen_1_1LevenbergMarquardt.html).

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"GradientTolerance"   | <tt>double</tt> | <tt>1.0e-8</tt> | The tolerance for the gradient norm. |
"FunctionTolerance"   | <tt>double</tt> | <tt>1.0e-8</tt> | The tolerance for the function norm. |
"ParameterTolerance"   | <tt>double</tt> | <tt>1.0e-8</tt> | The tolerance for the parameter norm. |
"MaximumFunctionEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of function evaluations. |
"MaximumJacobianEvaluations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of Jacobian evaluations. |
"MaximumIterations"   | <tt>std::size_t</tt> | <tt>1000</tt> | The maximum number of iterations. |
"Step"   | <tt>double</tt> | <tt>100.0</tt> | Step factor that bounds the maximum step size per iteration. |
*/
template<typename MatrixType, typename QRSolver>
class LevenbergMarquardt {
public:

  /// Information about whether or not the algorithm converged
  enum Convergence {
    /// Hit the maximum number of Jacobian evaluations
    FAILED_MAX_NUM_JACOBIAN_EVALS=-2,

    /// Hit the maximum number of function evaluations
    FAILED_MAX_NUM_COST_EVALS=-1,

    /// The algorithm is currently running
    CONTINUE_RUNNING = 0,

    /// Converged because the relative error is small
    CONVERGED_RELATIVE_ERROR_SMALL=1,

    /// Converged because the relative reduction is small
    CONVERGED_RELATIVE_REDUCTION_SMALL=2,

    /// Converged because both the relative error and the relative reduction is small
    CONVERGED_RELATIVE_ERROR_AND_REDUCTION_SMALL=3,

    /// Converged because the cost function is small
    CONVERGED_FUNCTION_SMALL=4,

    /// Converged because the step size is small
    CONVERGED_STEP_SMALL=5,

    /// Converged because the gradient is small
    CONVERGED_GRADIENT_SMALL=6,
  };

  /**
  @param[in] cost The cost function that we need to minimize
  */
  inline LevenbergMarquardt(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  cost(cost),
  gradTol(pt.get<double>("GradientTolerance", 1.0e-8)),
  funcTol(pt.get<double>("FunctionTolerance", 1.0e-8)),
  betaTol(pt.get<double>("ParameterTolerance", 1.0e-8)),
  maxEvals(pt.get<std::size_t>("MaximumFunctionEvaluations", 1000)),
  maxJacEvals(pt.get<std::size_t>("MaximumJacobianEvaluations", 1000)),
  maxIters(pt.get<std::size_t>("MaximumIterations", 250)),
  stepFactor(pt.get<double>("StepFactor", 100.0))
  {}

  virtual ~LevenbergMarquardt() = default;

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  */
  inline void Minimize(Eigen::VectorXd& beta) {
    Eigen::VectorXd costVec;
    Minimize(beta, costVec);
  }

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
  @param[in,out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The paramter values that minimize the cost function
  @param[out] costVec The evaluation of the cost function at the optimizal point
  */
  inline void Minimize(Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    assert(beta.size()==cost->inDim);

    // reset the algorithm parameters (e.g., function evaluation counters)
    ResetParameters();

    // evaluate the cost at the initial guess
    EvaluateCost(beta, costVec);

    double prevCost = costVec.dot(costVec);

    std::size_t iter = 0;
    double damping = 0.0;
    while( iter<maxIters ) {
      std::cout << std::endl << "iter: " << iter << std::endl;

      Convergence conv;
      double newCost;
      //Eigen::VectorXd newBeta = beta;
      std::tie(conv, newCost) = Iteration(iter, damping, beta, costVec);
      std::cout << "cost: " << newCost << std::endl;

      if( newCost<prevCost ) {
        //beta = newBeta;
        //prevCost = newCost;
        damping *= 0.75;
      } else {
        damping *= 2.0;
        damping = std::min(1.0, damping);
      }
      std::cout << "new damping: " << damping << std::endl;

      prevCost = newCost;

      if( conv<0 ) {
        //std::cout << "beta: " << beta.transpose() << std::endl;
        //std::cout << "cost: " << costVec.transpose() << std::endl;
        //std::cout << "CONV: " << conv << std::endl;
      }
      //assert(conv>=0);

      if( conv>0 ) { break; }
    }
  }

  /// The tolerance for convergence based on the gradient norm
  const double gradTol;

  /// The tolerance for convergence based on the function norm
  const double funcTol;

    /// The tolerance for convergence based on the parameter norm
  const double betaTol;

  /// The maximum number of cost function evaluations
  const std::size_t maxEvals;

  /// The maximum number of Jacobian evaluations
  const std::size_t maxJacEvals;

  /// The maximum number of iterations
  const std::size_t maxIters;

protected:

  virtual void AddScaledIdentity(double const scale, MatrixType& mat) const = 0;

private:

  /// Reset the parameters for the Levenberg Marquardt algorithm
  inline void ResetParameters() {
    numCostEvals = 0;
    numJacEvals = 0;
    scaling.resize(cost->inDim);

    lmPara = 0.0;
  }

  /// Perform one iteration of the Levenberg Marquardt algorithm
  /**
  @param[in,out] iter In: The current iteration number, Out: The incremeneted iterationnumber
  @param[in,out] beta In: The current parameter value, Out: The updated parameter value
  @param[in,out] costVec In: The cost given the current parameter value, Out: The cost at the next iteration
  */
  inline std::pair<Convergence, double> Iteration(std::size_t& iter, double const damping, Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    //std::cout << std::endl << std::endl << std::endl;
    ++iter;

    // evaluate the cost at the initial guess
    EvaluateCost(beta, costVec);
    //if( costVec.dot(costVec)<funcTol ) { return Convergence::CONVERGED_FUNCTION_SMALL; }

    // compute the Jacobian matrix
    MatrixType jac;
    Jacobian(beta, jac);
    std::cout << "grad norm: " << (jac.adjoint()*costVec).norm() << " tol: " << gradTol << std::endl;
    if( (jac.adjoint()*costVec).norm()<gradTol ) { return std::pair<Convergence, double>(Convergence::CONVERGED_GRADIENT_SMALL, costVec.dot(costVec)); }

    //std::cout << jac.adjoint()*costVec << std::endl;

    //if( costVec.dot(costVec)<funcTol ) { return Convergence::CONVERGED_FUNCTION_SMALL; }

    std::cout << "cost: " << costVec.dot(costVec) << std::endl;

    // form (q transpose)*costVec and store the first n components in qtCost.
    //Eigen::VectorXd QTcost = (qrfac.matrixQ().adjoint()*costVec).head(cost->inDim);
    Eigen::VectorXd QTcost = jac.transpose()*costVec;
    //std::cout << "Q^T cost: " << QTcost.transpose() << std::endl;

    // compute the QR factorization of the Jacobian matrix
    jac = jac.transpose()*jac;
    AddScaledIdentity(damping, jac);
    //for( std::size_t i=0; i<jac.cols(); ++i ) { jac.coeffRef(i, i) += damping; }
    //jac.makeCompressed();
    QRSolver qrfac(jac);
    assert(qrfac.info()==Eigen::Success);
    const MatrixType& matrixR = qrfac.matrixR();

    QTcost = qrfac.matrixQ().adjoint()*QTcost;

    Eigen::VectorXd stepDir(cost->inDim);
    const Eigen::Index rank = qrfac.rank();
    std::cout << "rank: " << rank << " of " << beta.size() << std::endl;
    stepDir.tail(cost->inDim-rank).setZero();

    stepDir.head(rank) = matrixR.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solve((QTcost).head(rank));
    //std::cout << "step dir: " << stepDir.transpose() << std::endl;
    stepDir = qrfac.colsPermutation()*stepDir;
    //std::cout << "step dir: " << stepDir.transpose() << std::endl;

    beta -= stepDir;

    //std::cout << "new beta: " << beta.transpose() << std::endl;


    //assert(false);

    return std::pair<Convergence, double>(Convergence::FAILED_MAX_NUM_COST_EVALS, costVec.dot(costVec));

    /*
    ++iter;

    double costNorm = costVec.stableNorm();

    double temp, temp1,temp2;
    double pnorm, betaNorm, fnorm1, actred, dirder, prered;
    eigen_assert(beta.size()==cost->inDim);

    temp = 0.0; betaNorm = 0.0;
    // compute the Jacobian matrix
    MatrixType jac;
    Jacobian(beta, jac);

    std::cout << "done jac" << std::endl;

    // temporary vectors
    Eigen::VectorXd tempVec1, tempVec2(cost->inDim), tempVec3, tempVec4;

    // compute the QR factorization of the Jacobian
    for( std::size_t j=0; j<cost->inDim; ++j ) { tempVec2(j) = jac.col(j).blueNorm(); }
    QRSolver qrfac(jac);
    assert(qrfac.info()==Eigen::Success);

    // make a copy of the first factor with the associated permutation
    const MatrixType& matrixR = qrfac.matrixR();
    const typename QRSolver::PermutationType& permutation = qrfac.colsPermutation();

    // on the first iteration, scale according to the norms of the columns of the initial jacobian
    if( iter==1 ) {
      for( Eigen::Index j=0; j<cost->inDim; ++j ) { scaling[j] = (std::abs(tempVec2[j])<1.0e-14)? 1.0 : tempVec2[j]; }

      // on the first iteration, calculate the norm of the scaled x and initialize the step bound step bound.
      betaNorm = scaling.cwiseProduct(beta).stableNorm();
      stepBound = stepFactor * betaNorm;
      if( std::abs(stepBound)<1.0e-14 ) { stepBound = stepFactor; }
    }

    // form (q transpose)*costVec and store the first n components in qtCost.
    tempVec4 = qrfac.matrixQ().adjoint() * costVec;
    Eigen::VectorXd qtCost = tempVec4.head(cost->inDim);

    std::cout << "HERE OKAY" << std::endl;

    // compute the norm of the scaled gradient.
    double gradNorm = 0.0;
    if( costNorm>1.0e-14 ) {
      for( Eigen::Index j = 0; j<cost->inDim; ++j ) {
        if( std::abs(tempVec2[permutation.indices()[j]])>1.0e-14 ) {
          gradNorm = std::max(gradNorm, abs( matrixR.col(j).head(j+1).dot(qtCost.head(j+1)/costNorm) / tempVec2[permutation.indices()[j]]));
        }
      }
    }

    // test for convergence of the gradient norm.
    if( gradNorm<gradTol) { return CONVERGED_GRADIENT_SMALL; }

    // test for convergence of the gradient norm.
    if( costNorm<funcTol) { return CONVERGED_FUNCTION_SMALL; }

    // rescale if necessary
    std::cout << "TV2: " << tempVec2.transpose() << std::endl;
    std::cout << "scaling: " << scaling.transpose() << std::endl;
    scaling = scaling.cwiseMax(tempVec2);

    double ratio = 0.0;
    Convergence conv = Convergence::CONTINUE_RUNNING;
    while( ratio<1.0e-4 && conv==Convergence::CONTINUE_RUNNING ) {
      std::cout << "IN LOOP" << std::endl;
      // determine the levenberg-marquardt parameter
      ComputeParameters(qrfac, scaling, qtCost, stepBound, tempVec1);

      std::cout << "OKAY" << std::endl;

      // store the direction p and beta + p. calculate the norm of p
      tempVec1 = -tempVec1;
      tempVec2 = beta + tempVec1;
      pnorm = scaling.cwiseProduct(tempVec1).stableNorm();

      // on the first iteration, adjust the initial step bound
      if( iter==1 ) { stepBound = std::min(stepBound, pnorm); }

      // evaluate the function at x + p and calculate its norm
      EvaluateCost(tempVec2, tempVec4);
      fnorm1 = tempVec4.stableNorm();

      // compute the scaled actual reduction
      actred = -1.0;
      if( 0.1*fnorm1<costNorm ) { actred = 1.0-Eigen::numext::abs2(fnorm1 / costNorm); }

      // compute the scaled predicted reduction and the scaled directional derivative.
      tempVec3 = matrixR.template triangularView<Eigen::Upper>() * (permutation.inverse() *tempVec1);
      temp1 = Eigen::numext::abs2(tempVec3.stableNorm() / costNorm);
      temp2 = Eigen::numext::abs2(sqrt(lmPara) * pnorm / costNorm);
      prered = temp1 + temp2 / 0.5;
      dirder = -temp1 - temp2;

      // compute the ratio of the actual to the predicted reduction.
      ratio = 0.0;
      if( prered!=0.0 ) { ratio = actred / prered; }

      // update the step bound
      if( ratio<=0.25 ) {
        if( actred>=0.0 ) { temp = 0.5; }
        if( actred < 0.0 ) { temp = 0.5 * dirder / (dirder + 0.5 * actred); }
        if( 0.1*fnorm1>=costNorm || temp<0.1 ) { temp = 0.1; }
        // Computing MIN
        stepBound = temp * std::min(stepBound, pnorm / 0.1);
        lmPara /= temp;
      } else if( !(lmPara!=0.0 && ratio<0.75) ) {
        stepBound = pnorm / 0.5;
        lmPara = 0.5 * lmPara;
      }

      // test for successful iteration
      if( ratio >= 1.0e-4 ) {
        // successful iteration. update x, costVec, and their norms
        beta = tempVec2;
        tempVec2 = scaling.cwiseProduct(beta);
        costVec = tempVec4;
        betaNorm = tempVec2.stableNorm();
        costNorm = fnorm1;
      }

      // check convergence
      conv = CheckConvergence(std::abs(actred), prered, ratio, stepBound, betaNorm, gradNorm, numCostEvals, numJacEvals);
    }

    return conv;*/
  }

  /// Evaluate the cost function
  /**
  @param[in] beta The parameter value
  @param[out] costVec The \f$i^{th}\f$ componenent is the evaluation of the function \f$f_i(\boldsymbol{\beta})\f$
  \return The sum of the sub-cost functions
  */
  inline void EvaluateCost(Eigen::VectorXd const& beta, Eigen::VectorXd& costVec) {
    costVec = cost->Cost(beta);
    ++numCostEvals;
  }

  /// Compute the Jacobian matrix
  /**
  Also, increment the counter for the number of times we needed to compute the Jacobian matrix
  @param[in] beta The parameter value
  @param[out] jac The Jacobian matrix
  */
  inline void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) {
    cost->Jacobian(beta, jac);
    ++numJacEvals;
  }

  /**
  @param[in] qr The QR factorization of the Jacobian matrix
  @param[in] scaling The scaling for the step
  @param[in] qtb The first \f$n\f$ components of \f$Q^T c\f$, where \f$Q\f$ is from the QR factorization and \f$c\f$ is the cost vector
  @param[in] stepBound The bound for the step size
  @param[out] stepDir The step direction
  */
  inline void ComputeParameters(QRSolver const& qr, Eigen::VectorXd const& scaling, Eigen::VectorXd const& qtb, double stepBound, Eigen::VectorXd& stepDir) {
    typedef typename QRSolver::Scalar Scalar;

    // Local variables
    Eigen::Index j;
    Scalar fp;
    Scalar parc;
    Scalar temp;
    Scalar gnorm;
    Scalar dxnorm;

    // make a copy of the triangular factor; this copy is modified during call the qrsolv
    MatrixType matrixR = qr.matrixR();

    const Scalar dwarf = std::numeric_limits<Scalar>::min();
    const Eigen::Index n = qr.matrixR().cols();
    eigen_assert(n==scaling.size());
    eigen_assert(n==qtb.size());

    Eigen::VectorXd  tempVec1, tempVec2;

    // compute and store in stepDir the gauss-newton direction. if the jacobian is rank-deficient, obtain a least squares solution.

    const Eigen::Index rank = qr.rank();
    tempVec1 = qtb;
    tempVec1.tail(n-rank).setZero();
    tempVec1.head(rank) = matrixR.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solve(qtb.head(rank));
    stepDir = qr.colsPermutation()*tempVec1;

    // initialize the iteration counter
    Eigen::Index iter = 0;

    // evaluate the function at the origin, and test for acceptance of the gauss-newton direction
    tempVec2 = scaling.cwiseProduct(stepDir);
    dxnorm = tempVec2.blueNorm();
    fp = dxnorm - stepBound;
    if( fp<=Scalar(0.1)*stepBound ) {
      lmPara = 0;
      return;
    }

    // if the jacobian is not rank deficient, the newton step provides a lower bound, lower, for the zero of the function; otherwise set this bound to zero
    double lower = 0.0;
    if( rank==n ) {
      tempVec1 = qr.colsPermutation().inverse()*scaling.cwiseProduct(tempVec2)/dxnorm;
      matrixR.topLeftCorner(n, n).transpose().template triangularView<Eigen::Lower>().solveInPlace(tempVec1);
      temp = tempVec1.blueNorm();
      lower = fp / stepBound / temp / temp;
    }

    // calculate an upper bound, upper, for the zero of the function.
    for( j=0; j<n; ++j ) { tempVec1[j] = matrixR.col(j).head(j+1).dot(qtb.head(j+1)) / scaling[qr.colsPermutation().indices()(j)]; }

    gnorm = tempVec1.stableNorm();
    double upper = gnorm / stepBound;
    if( upper==0.0 ) { upper = dwarf / (std::min)(stepBound,Scalar(0.1)); }

    // if the input par lies outside of the interval (lower,upper), set par to the closer endpoint.
    lmPara = std::max(lmPara, lower);
    lmPara = std::min(lmPara, upper);
    if( lmPara==0.0 ) { lmPara = gnorm / dxnorm; }

    // beginning of an iteration
    while( true ) {
      ++iter;

      // evaluate the function at the current value of par.
      if( std::abs(lmPara)<1.0e-14 ) { lmPara = std::max(dwarf, Scalar(.001)*upper); }
      tempVec1 = std::sqrt(lmPara)*scaling;

      Eigen::VectorXd sdiag(n);
      UpdateQR(qr, matrixR, qr.colsPermutation(), tempVec1, qtb, stepDir, sdiag);

      tempVec2 = scaling.cwiseProduct(stepDir);
      dxnorm = tempVec2.blueNorm();
      temp = fp;
      fp = dxnorm - stepBound;

      // if the function is small enough, accept the current value of lmPara; also test for the exceptional cases where lower is zero or the number of iterations has reached 10
      if( abs(fp)<=Scalar(0.1)*stepBound || (lower==0.0 && fp<=temp && temp<0.0) || iter==10 ) { break; }

      // compute the newton correction
      tempVec1 = qr.colsPermutation().inverse()*scaling.cwiseProduct(tempVec2/dxnorm);
      // we could almost use this here, but the diagonal is outside qr, in sdiag[]
      for( j=0; j<n; ++j ) {
        tempVec1[j] /= sdiag[j];
        temp = tempVec1[j];
        for( Eigen::Index i = j+1; i<n; ++i ) { tempVec1[i] -= matrixR.coeff(i,j)*temp; }
      }
      temp = tempVec1.blueNorm();
      parc = fp / stepBound / temp / temp;

      // depending on the sign of the function, update lower or upper
      if( fp>0.0 ) { lower = std::max(lower, lmPara); }
      if( fp<0.0 ) { upper = std::min(upper, lmPara); }

      // compute an improved estimate for lmPara
      lmPara = std::max(lower, lmPara+parc);
    }

    if( iter==0 ) { lmPara = 0.0; }
  }

  /// Update the QR decomposition (dense version)
  /**
  @param[in, out] matrixR The upper trianguar part of the QR decomposition
  @param[in] permutation The permutation from the QR decomposition
  @param[in] diag The scaling vector
  @param[in] qtb The matrix Q transpose times the cost
  @param[in] stepDir The step direction
  @param[in] sdiag The weighted diagonal
  */
  template <typename Scalar,int Rows, int Cols, typename PermIndex>
  void UpdateQR(
    QRSolver const& qr,
    Eigen::Matrix<Scalar, Rows, Cols>& matrixR, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, PermIndex> const& permutation, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const& diag, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const& qtb, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& stepDir, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& sdiag)
  {
    // local variables
    Eigen::Index i, j, k;
    Scalar temp;
    Eigen::Index n = matrixR.cols();
    Eigen::Matrix<Scalar,Eigen::Dynamic,1>  wa(n);
    Eigen::JacobiRotation<Scalar> givens;

    // the following will only change the lower triangular part of s, including the diagonal, though the diagonal is restored afterward

    // copy r and (q transpose)*b to preserve input and initialize s; in particular, save the diagonal elements of r in stepDir
    stepDir = matrixR.diagonal();
    wa = qtb;

    matrixR.topLeftCorner(n,n).template triangularView<Eigen::StrictlyLower>() = matrixR.topLeftCorner(n,n).transpose();

    // eliminate the diagonal matrix d using a givens rotation
    for( j=0; j<n; ++j ) {
      // prepare the row of d to be eliminated, locating the diagonal element using p from the qr factorization
      const PermIndex l = permutation.indices()(j);
      if( diag[l]==0.0 ) { break; }
      sdiag.tail(n-j).setZero();
      sdiag[j] = diag[l];

      // the transformations to eliminate the row of d modify only a single element of (q transpose)*b beyond the first n, which is initially zero
      Scalar qtbpj = 0.;
      for( k=j; k<n; ++k ) {
        // determine a givens rotation which eliminates the appropriate element in the current row of d
        givens.makeGivens(-matrixR(k,k), sdiag[k]);

        // compute the modified diagonal element of r and the modified element of ((q transpose)*b,0)
        matrixR(k,k) = givens.c() * matrixR(k,k) + givens.s() * sdiag[k];
        temp = givens.c() * wa[k] + givens.s() * qtbpj;
        qtbpj = -givens.s() * wa[k] + givens.c() * qtbpj;
        wa[k] = temp;

        // accumulate the transformation in the row of s
        for( i=k+1; i<n; ++i ) {
          temp = givens.c() * matrixR(i,k) + givens.s() * sdiag[i];
          sdiag[i] = -givens.s() * matrixR(i,k) + givens.c() * sdiag[i];
          matrixR(i,k) = temp;
        }
      }
    }

    // solve the triangular system for z. if the system is singular, then obtain a least squares solution
    Eigen::Index nsing;
    for( nsing=0; nsing<n && std::abs(sdiag[nsing])>1.0e-14; nsing++ ) {}

    wa.tail(n-nsing).setZero();
    matrixR.topLeftCorner(nsing, nsing).transpose().template triangularView<Eigen::Upper>().solveInPlace(wa.head(nsing));

    // restore
    sdiag = matrixR.diagonal();
    matrixR.diagonal() = stepDir;

    // permute the components of z back to components of stepDir
    stepDir = permutation*wa;
  }

  /// Update the QR decomposition (sparse version)
  /**
  @param[in, out] matrixR The upper trianguar part of the QR decomposition
  @param[in] permutation The permutation from the QR decomposition
  @param[in] diag The scaling vector
  @param[in] qtb The matrix Q transpose times the cost
  @param[in] stepDir The step direction
  @param[in] sdiag The weighted diagonal
  */
  template <typename Scalar, int _Options, typename Index>
  void UpdateQR(
    QRSolver const& qr,
    Eigen::SparseMatrix<Scalar, _Options, Index>& matrixR, Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> const& permutation, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const& diag, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const& qtb, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& stepDir, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& sdiag) {
  // /ocal variables
  typedef Eigen::SparseMatrix<Scalar,Eigen::RowMajor,Eigen::Index> FactorType;
  Eigen::Index i, j, k, l;
  Scalar temp;
  Eigen::Index n = matrixR.cols();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> wa(n);
  Eigen::JacobiRotation<Scalar> givens;

  // the following will only change the lower triangular part of s, including the diagonal, though the diagonal is restored afterward

  // copy r and (q transpose)*b to preserve input and initialize R
  wa = qtb;
  FactorType R(matrixR);

  // eliminate the diagonal matrix d using a givens rotation
  for( j=0; j<n; ++j ) {
    // prepare the row of d to be eliminated, locating the diagonal element using p from the qr factorization
    l = permutation.indices()(j);

    if( std::abs(diag(l))<1.0e-14 ) { break; }
    sdiag.tail(n-j).setZero();
    sdiag[j] = diag[l];

      // the transformations to eliminate the row of d modify only a single element of (q transpose)*b beyond the first n, which is initially zero.

      Scalar qtbpj = 0;
      // browse the nonzero elements of row j of the upper triangular s
      for( k=j; k<n; ++k ) {
        typename FactorType::InnerIterator itk(R,k);
        for( ; itk; ++itk ) {
          if( itk.index()<k) { continue; } else { break; }
        }

        // at this point, we have the diagonal element R(k,k) determine a givens rotation which eliminates the appropriate element in the current row of d
        givens.makeGivens(-itk.value(), sdiag(k));

        // compute the modified diagonal element of r and the modified element of ((q transpose)*b,0).
        itk.valueRef() = givens.c() * itk.value() + givens.s() * sdiag(k);
        temp = givens.c() * wa(k) + givens.s() * qtbpj;
        qtbpj = -givens.s() * wa(k) + givens.c() * qtbpj;
        wa(k) = temp;

        // accumulate the transformation in the remaining k row/column of R
        for( ++itk; itk; ++itk ) {
          i = itk.index();
          temp = givens.c() *  itk.value() + givens.s() * sdiag(i);
          sdiag(i) = -givens.s() * itk.value() + givens.c() * sdiag(i);
          itk.valueRef() = temp;
        }
      }
    }

    // Solve the triangular system for z. If the system is singular, then obtain a least squares solution
    Index nsing;
    //for( nsing=0; nsing<n && std::abs(sdiag(nsing))>1.0e-14 && std::abs(R.coeff(nsing, nsing))>1.0e-14; nsing++) {}
    for( nsing=0; nsing<n && std::abs(sdiag(nsing))>1.0e-14; nsing++) {}

    //std::cout << R.topLeftCorner(nsing,nsing) << std::endl << std::endl;

    //std::cout << R.topLeftCorner(nsing,nsing).template triangularView<Eigen::Upper>().solve(wa.head(nsing)) << std::endl;
    //std::cout << "HIOGHROW" << std::endl;

    wa.tail(n-nsing).setZero();
    wa.head(nsing) = R.topLeftCorner(nsing,nsing).template triangularView<Eigen::Upper>().solve(wa.head(nsing));

    sdiag = R.diagonal();
    // permute the components of z back to components of stepDir
    stepDir = permutation*wa;
  }

  Convergence CheckConvergence(double const actualReduction, double const predictedReduction, double const reductionRatio, double const stepBound, double const betaNorm, double const gradNorm, std::size_t const numCostEvals, std::size_t const numJacEvals) const {
    // check for relative convergence
    if( actualReduction<funcTol && predictedReduction<funcTol && 0.5*reductionRatio<1.0 && stepBound<betaTol*betaNorm ) { return Convergence::CONVERGED_RELATIVE_ERROR_AND_REDUCTION_SMALL; }

    if( actualReduction<funcTol && predictedReduction<funcTol && 0.5*reductionRatio<1.0 ) { return Convergence::CONVERGED_RELATIVE_REDUCTION_SMALL; }

    if( stepBound<betaTol*betaNorm ) { return Convergence::CONVERGED_RELATIVE_ERROR_SMALL; }

    // check the number of iterations
    if( numCostEvals>maxEvals ) { return Convergence::FAILED_MAX_NUM_COST_EVALS; }
    if( numJacEvals>maxJacEvals ) { return Convergence::FAILED_MAX_NUM_JACOBIAN_EVALS; }

    // check aboslute convergence
    if( actualReduction<Eigen::NumTraits<double>::epsilon() && predictedReduction<Eigen::NumTraits<double>::epsilon() ) { return Convergence::CONVERGED_FUNCTION_SMALL; }
    if( stepBound<Eigen::NumTraits<double>::epsilon()*betaNorm ) { return Convergence::CONVERGED_STEP_SMALL; }
    if( gradNorm<Eigen::NumTraits<double>::epsilon() ) { return Convergence::CONVERGED_GRADIENT_SMALL; }

    return Convergence::CONTINUE_RUNNING;
  }

  /// The scaling used in the Levenberg Marquardt algorithm
  Eigen::VectorXd scaling;

  /// The cost function that we need to minimize
  std::shared_ptr<CostFunction<MatrixType> > cost;

  /// The Levenberg Marquardt parameter
  double lmPara = 0.0;

  /// The number of cost function evaluations
  std::size_t numCostEvals = 0;

  /// The number of Jacobian evaluations
  std::size_t numJacEvals = 0;

  /// The factor that determines the step bound for each iteration
  const double stepFactor;

  /// The step bound for each iteration
  double stepBound;
};

class DenseLevenbergMarquardt : public LevenbergMarquardt<Eigen::MatrixXd, Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > {
public:
  /**
  @param[in] cost The cost function that we need to minimize
  */
  inline DenseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::MatrixXd> > const& cost, boost::property_tree::ptree const& pt) : LevenbergMarquardt<Eigen::MatrixXd, Eigen::ColPivHouseholderQR<Eigen::MatrixXd> >(cost, pt) {}

  virtual ~DenseLevenbergMarquardt() = default;

protected:

  inline virtual void AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const override { mat += scale*Eigen::MatrixXd::Identity(mat.rows(), mat.cols()); }

private:
};

class SparseLevenbergMarquardt : public LevenbergMarquardt<Eigen::SparseMatrix<double>, Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > > {
public:
  /**
  @param[in] cost The cost function that we need to minimize
  */
  inline SparseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::SparseMatrix<double> > > const& cost, boost::property_tree::ptree const& pt) : LevenbergMarquardt<Eigen::SparseMatrix<double>, Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > >(cost, pt) {}

  virtual ~SparseLevenbergMarquardt() = default;

protected:

  inline virtual void AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const override {
    for( std::size_t i=0; i<std::min(mat.rows(), mat.cols()); ++i ) { mat.coeffRef(i, i) += scale; }
    mat.makeCompressed();
  }

private:
};

} // namespace clf

#endif
