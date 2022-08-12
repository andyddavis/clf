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

  /// Map to a hypercube \f$[-1, 1]^d\f$
  /**
     @param[in] x A point in the domain \f$x \in \Omega\f$
     \return A point in the hypercube \f$[-1, 1]^d\f$
   */
  virtual Eigen::VectorXd MapToHypercube(Eigen::VectorXd const& x) const;

  /// An exception to be thrown if none of the proposed samples are valid
  class SampleFailure : public std::logic_error {
  public:

    /**
       @param[in] nproposals The number of proposals before we gave up
     */
    SampleFailure(std::size_t const nproposals);

    virtual ~SampleFailure() = default;
  private:
  };

  /// Generate a sample in the domain
  /**
     \return A point in the domain
   */
  Eigen::VectorXd Sample();

  /// Compute the distance between two points in the domain
  /**
     Defaults to the 2-norm.
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

  /// The parameters for this domain
  std::shared_ptr<const Parameters> para;
  
private:

  /// A domain that is a super set of this domain
  /**
     If this is the null pointer, then the domain has no super set and we don't need to check to make sure new points are contained inside this domain.
   */
  std::shared_ptr<Domain> super;

  /// The maximum number of samples from the domain that clf::Domain::Sample() will propose before crashing. Proposed samples are rejected if they are not in the superset.
  inline static std::size_t maxProposedSamps_DEFAULT = 10000;
  
};
  
} // namespace clf

#endif
