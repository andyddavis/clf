#ifndef TESTMODELS_HPP_
#define TESTMODELS_HPP_

#include "clf/Model.hpp"

namespace clf {
namespace tests {

/// Implement the model \f$\mathcal{L}(u) = [u_0, u_1^2 + u_1]^{\top} = [\sin{(2 \pi x_0)} \cos{(\pi x_1)} + \cos{(x_0)}, x_0 x_1]^{\top}\f$
class TwoDimensionalAlgebraicModel : public Model {
public:

  inline TwoDimensionalAlgebraicModel(boost::property_tree::ptree const& pt) : Model(pt) {}

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
  \return The evaluation of \f$mathcal{L}(u)\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output(1) += output(1)*output(1);

    return output;
  }

  /// Implement the Jacobian \f$\nabla_{p} \mathcal{L}(u)\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients \f$p\f$
  @param[in] bases the basis function \f$\phi\f$
  \return The evaluation of \f$mathcal{L}(u)\f$
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

    return jac;
  }

private:
};

} // namespace tests
} // namespace clf

#endif
