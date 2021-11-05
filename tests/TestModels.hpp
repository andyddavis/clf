#ifndef TESTMODELS_HPP_
#define TESTMODELS_HPP_

#include "clf/Model.hpp"

namespace clf {
namespace tests {

/// Implement the model \f$\mathcal{L}(u) = [u_0, u_1^2 + u_1]^{\top} = [\sin{(2 \pi x_0)} \cos{(\pi x_1)} + \cos{(x_0)}, x_0 x_1]^{\top}\f$
class TwoDimensionalAlgebraicModel : public Model {
public:

  /// Construct the model
  /**
  @param[in] pt Options for the model---both the input and output should be \f$2\f$
  */
  inline TwoDimensionalAlgebraicModel(boost::property_tree::ptree const& pt) : Model(pt) {
    assert(inputDimension==2);
    assert(outputDimension==2);
  }

  virtual ~TwoDimensionalAlgebraicModel() = default;
protected:

  /// Implement the right hand side \f$f(x) = [\sin{(2 \pi x_0)} \cos{(\pi x_1)} + \cos{(x_0)}, x_0 x_1]^{\top}\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override {
    return Eigen::Vector2d(
      std::sin(2.0*M_PI*x(0))*std::cos(M_PI*x(1)) + std::cos(x(0)),
      x.prod()
    );
  }

  /// Implement the operator \f$\mathcal{L}(u) = [u_0, u_1^2 + u_1]^{\top} = [\phi_{\hat{x}}^{(0)}(x) \cdot p_0, (\phi_{\hat{x}}^{(1)}(x) \cdot p_1)^2 + \phi_{\hat{x}}^{(1)}(x) \cdot p_1]^{\top}\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients \f$p\f$
  @param[in] bases the basis function \f$\phi\f$
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output(1) += output(1)*output(1);

    return output;
  }

  /// Implement the Jacobian \f$\nabla_{p} \mathcal{L}(u)\f$
  /**
  The Jacobian of the operator \f$\mathcal{L}(u)\f$ with respect to the basis function coefficients is
  \f{equation*}{
     \nabla_{p} \mathcal{L}(u) = \nabla_{p} \mathcal{L}(\Phi_{x^{\prime}}(x) p) = \left[ \begin{array}{ccc|ccc}
        --- & (\boldsymbol{\phi}_{x^{\prime}}^{(0)}(x))^{\top} & --- & --- & 0 & --- \\
        --- & 0 & --- & --- & 2 (\phi_{\hat{x}}^{(1)}(x) \cdot p_1) (\boldsymbol{\phi}_{x^{\prime}}^{(1)}(x))^{\top} + (\boldsymbol{\phi}_{x^{\prime}}^{(1)}(x))^{\top} & ---
     \end{array} \right].
  \f}

  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients \f$p\f$
  @param[in] bases the basis function \f$\phi\f$
  \return The evaluation of \f$\nabla_{p} \mathcal{L}(u)\f$
  */
  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);

      if( i==1 ) {
        jac.row(i).segment(ind, basisSize) = phi + 2.0*phi.dot(coefficients.segment(ind, basisSize))*phi;
      } else {
        jac.row(i).segment(ind, basisSize) = phi;
      }

      ind += basisSize;
    }

    /*const Eigen::MatrixXd jacFD = OperatorJacobianByFD(x, coefficients, bases);
    //std::cout << "FD jac: " << OperatorJacobianByFD(x, coefficients, bases) << std::endl;
    //std::cout << "jac: " << jac << std::endl;
    std::cout << "model jac FD check: " << (jacFD-jac).norm() << std::endl;
    std::cout << std::endl;
    return jacFD;*/

    return jac;
  }

  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    std::vector<Eigen::MatrixXd> hess(2);
    hess[0] = Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size());

    const Eigen::VectorXd phi = bases[1]->EvaluateBasisFunctions(x);
    hess[1] = Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size());
    hess[1].block(bases[0]->NumBasisFunctions(), bases[0]->NumBasisFunctions(), bases[1]->NumBasisFunctions(), bases[1]->NumBasisFunctions()) = 2.0*phi*phi.transpose();

    return hess;
  }

private:
};

/// Implement the model \f$\mathcal{L}(u) = u^2\f$, where the squared operator is applied component-wise.
class ComponentWiseSquaredModel : public Model {
public:

  /// Construct the model
  /**
  @param[in] pt Options for the model
  */
  inline ComponentWiseSquaredModel(boost::property_tree::ptree const& pt) : Model(pt) {}

  virtual ~ComponentWiseSquaredModel() = default;

protected:

  /// Evaluate the right hand side function \f$f(x) = (x \cdot x) \mathbf{1}\f$, where \f$\mathbf{1}\f$ is a vector of ones of the appropriate length
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension, x.dot(x)); }

  /// Implement the model \f$\mathcal{L}(u) = u^2\f$, where the squared operator is applied component-wise.
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients that multiply the basis functions 
  @param[in] bases The basis functions 
  \return The operator evaluation \f$\mathcal{L}(u) = u^2 = (\Phi_{\hat{x}}(x) p)^2\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output = output.array()*output.array();

    return output;
  }

  /// Compute the Jacobian with respect to the coefficients
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients that multiply the basis functions 
  @param[in] bases The basis functions 
  \return The operator Jacobian \f$\nabla_{p} \mathcal{L}(u)\f$
  */
  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);
      jac.row(i).segment(ind, basisSize) = phi.dot(coefficients.segment(ind, basisSize))*phi;
      ind += basisSize;
    }

    return 2.0*jac;
  }

  /// Compute the Hessian of the operator with respect to the coefficients
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size()));

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      for( std::size_t k=0; k<phi.size(); ++k ) {
        for( std::size_t j=0; j<phi.size(); ++j ) {
          hess[i](ind+k, ind+j) = 2.0*phi(k)*phi(j);
        }
      }
      ind += bases[i]->NumBasisFunctions();
    }

    return hess;
  }

private:
};

/// Intentially return the wrong number of outputs to test the exceptions
class TestVectorValuedImplementationModelWrongOutputDim : public Model {
public:

  /// Construct the model
  /**
  @param[in] pt Options for the model
  */ 
 inline TestVectorValuedImplementationModelWrongOutputDim(boost::property_tree::ptree const& pt) : Model(pt) {}

  virtual ~TestVectorValuedImplementationModelWrongOutputDim() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension+1, x.prod()); }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return Eigen::VectorXd::Constant(outputDimension+1, x.prod()); }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return Eigen::MatrixXd(outputDimension+1, coefficients.size()+1); }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return std::vector<Eigen::MatrixXd>(outputDimension+1); }

private:
};

/// Intentailly implement the Hessian with the wrong number of outputs
class TestHessianImplementationModelWrongOutputDim : public Model {
public:
  /// Construct the model
  /**
  @param[in] pt Options for the model
  */
  inline TestHessianImplementationModelWrongOutputDim(boost::property_tree::ptree const& pt) : Model(pt) {}

  virtual ~TestHessianImplementationModelWrongOutputDim() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    return IdentityOperator(x, coefficients, bases);
  }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of the Hessian of \f$\mathcal{L}(u)\f$
  */
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd(coefficients.size(), coefficients.size()));
    hess[0] = Eigen::MatrixXd(coefficients.size()+1, coefficients.size()+1);
    return hess;
  }

private:
};

/// Implement the differential operator so that the \f$i^{th}\f$ output is \f$\mathcal{L}_i(u) = u_i \sum_{j=1}^{n} \frac{\partial u_i}{\partial x_j}\f$
class TestDifferentialOperatorImplementationModel : public Model {
public:

  /// Construct the model
  /**
  @param[in] pt Options for the model
  */
  inline TestDifferentialOperatorImplementationModel(boost::property_tree::ptree const& pt) : Model(pt) {}

  virtual ~TestDifferentialOperatorImplementationModel() = default;

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override { return x.prod(); }

  /// The operator so that the \f$i^{th}\f$ output is \f$\mathcal{L}_i(u) = u_i \sum_{j=1}^{n} \frac{\partial u_i}{\partial x_j}\f$
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    const Eigen::VectorXd u = IdentityOperator(x, coefficients, bases);

    Eigen::VectorXd output = Eigen::VectorXd::Zero(outputDimension);
    for( std::size_t out=0; out<outputDimension; ++out ) {
      for( std::size_t in=0; in<inputDimension; ++in ) {
        output(out) += u(out)*FunctionDerivative(x, coefficients, bases, out, in, 1);
      }
    }

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      const std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);

      const Eigen::Map<const Eigen::VectorXd> coeffs(&coefficients[ind], basisSize);
      for( std::size_t p=0; p<inputDimension; ++p ) {
        const Eigen::VectorXd dphidx = bases[i]->EvaluateBasisFunctionDerivatives(x, p, 1);

        for( std::size_t j=0; j<basisSize; ++j ) {
          for( std::size_t s=0; s<basisSize; ++s ) {
            jac(i, ind+j) += (phi(j)*dphidx(s) + phi(s)*dphidx(j))*coeffs(s);
          }
        }
      }
      ind += basisSize;
    }

    return jac;
  }

  /// Compute the true Hessian of the operator with respect to the coefficients
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size()));

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      const std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);

      const Eigen::Map<const Eigen::VectorXd> coeffs(&coefficients[ind], basisSize);
      for( std::size_t p=0; p<inputDimension; ++p ) {
        const Eigen::VectorXd dphidx = bases[i]->EvaluateBasisFunctionDerivatives(x, p, 1);

        for( std::size_t j=0; j<basisSize; ++j ) {
          for( std::size_t s=0; s<basisSize; ++s ) {
            hess[i](ind+j, ind+s) += phi(j)*dphidx(s) + phi(s)*dphidx(j);
          }
        }
      }
      ind += basisSize;
    }

    return hess;
  }

private:
};

} // namespace tests
} // namespace clf

#endif
