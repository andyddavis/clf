#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <memory>

#include <Eigen/Core>

#include "clf/Parameters.hpp"

namespace clf {

/// Define a domain \f$\Omega \subseteq \mathbb{R}^{d}\f$
/**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "MaximumProposedSamples"   | <tt>std::size_t</tt> | <tt>10000</tt> | The maximum number of samples from the domain that clf::Domain::Sample() will propose before crashing. Proposed samples are rejected if they are not in the superset. See clf::Domain::maxProposedSamps_DEFAULT |
   "DeltaFD"   | <tt>double</tt> | <tt>1.0e-2</tt> | The step size for the finite difference approximation (see Domain::deltaFD_DEFAULT). |
   "OrderFD"   | <tt>std::size_t</tt> | <tt>8</tt> | The accuracy order for the finite difference approximation (see Domain::orderFD_DEFAULT). The options are \f$2\f$, \f$4\f$, \f$6\f$, and \f$8\f$. |
*/
class Domain {
public:
  
  /**
     @param[in] dim The dimension \f$d\f$
     @param[in] para The parameters for this domain
   */
  Domain(std::size_t const dim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~Domain() = default;

  /// Set another domain as a super set of this domain
  /**
     @param[in] supset The super set
   */
  void SetSuperset(std::shared_ptr<Domain> const& supset);

  /// Is a point inside the domain?
  /**
     @param[in] x The point
     \return <tt>true</tt>: Then \f$x \in \Omega\f$, <tt>false</tt>: Then \f$x \notin \Omega\f$
  */
  bool Inside(Eigen::VectorXd const& x) const;

  /// Diagonal, linear map to a subset of the hypercube \f$[-1, 1]^d\f$
  /**
     Define a projection \f$F: \Omega \mapsto [-1,1]^d\f$ such that \f$F(\Omega) \subseteq [-1,1]^d\f$. We map to the hypercube \f$[-1, 1]^d\f$ because that is where the clf::LegendrePolynomials are defined. Note that \f$F(\Omega)\f$ is a <em>subset</em> of \f$[-1,1]^d\f$. This because we assume this map is linear; dealing with a nonlinear map in higher dimensions leads to very complicated formulas for the derivative.

     This is defined by a diagonal matrix stored in clf::Domain::diagonalMap. If this is <tt>std::nullopt</tt> then this map is the identity.
     @param[in] x A point in the domain \f$x \in \Omega\f$
     \return A point in the hypercube \f$[-1, 1]^d\f$
   */
  virtual Eigen::VectorXd MapToHypercube(Eigen::VectorXd const& x) const;

  /// The Jacobian of the map to a subset of the hypercube \f$[-1, 1]\f$
  /**
     The map is diagonal, so return a vector (the diagonal)
     \return The Jacobian of the map to a subset of the hypercube \f$[-1, 1]\f$
   */
  Eigen::VectorXd MapToHypercubeJacobian() const;

  /// An exception to be thrown if none of the proposed samples are valid
  class SampleFailure : public std::logic_error {
  public:

    /**
       @param[in] funcName The function that triggered this exception
       @param[in] nproposals The number of proposals before we gave up
     */
    SampleFailure(std::string const& funcName, std::size_t const nproposals);

    /**
       @param[in] error A description of what went wrong with the sampling
     */
    SampleFailure(std::string const& error);

    virtual ~SampleFailure() = default;
  private:
  };

  /// Generate a sample in the domain
  /**
     \return A point in the domain
   */
  Eigen::VectorXd Sample();

  /// Generate a sample on the boundary of the domain
  /**
     \return First: A point on the domain boundary, Second: The outward pointing normal vector
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> SampleBoundary();

/// Compute the distance between two points in the domain
  /**
     If the domain has a super-set then use the super-set distance by default. Otherwise, default to the 2-norm. 
     @param[in] x1 The first point
     @param[in] x2 The second point
     \return The distance between the two points
   */
  virtual double Distance(Eigen::VectorXd const& x1, Eigen::VectorXd const& x2) const;

  /// The dimension of the domain \f$d\f$
  const std::size_t dim;

protected:

  /// Is a point inside the domain?
  /**
     @param[in] x The point
     \return <tt>true</tt>: Then \f$x \in \Omega\f$, <tt>false</tt>: Then \f$x \notin \Omega\f$
  */
  virtual bool CheckInside(Eigen::VectorXd const& x) const;

  /// Generate a sample in the domain
  /**
     \return A point in the domain
   */
  virtual Eigen::VectorXd ProposeSample();

  /// Generate a sample on the domain boundary
  /**
     \return First: A point on the domain boundary, Second: The outward pointing normal vector
   */
  virtual std::pair<Eigen::VectorXd, Eigen::VectorXd> ProposeBoundarySample();

  /// The parameters for this domain
  std::shared_ptr<const Parameters> para;

  /// A linear map to a subset of the hypercube \f$[-1, 1]\f$ defined by a diagonal matrix
  std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd> > map;

  /// A domain that is a super set of this domain
  /**
     If this is the null pointer, then the domain has no super set and we don't need to check to make sure new points are contained inside this domain.
   */
  std::shared_ptr<Domain> super;
  
private:

  /// The maximum number of samples from the domain that clf::Domain::Sample() will propose before crashing. Proposed samples are rejected if they are not in the superset.
  inline static std::size_t maxProposedSamps_DEFAULT = 10000;

  /// The default value for the finite diference delta
  inline static double deltaFD_DEFAULT = 1.0e-2;

  /// The default value for the finite diference order
  inline static std::size_t orderFD_DEFAULT = 8;
  
};
  
} // namespace clf

#endif
