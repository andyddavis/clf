#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <Eigen/Core>

namespace clf {

/// Define a domain \f$\Omega \subseteq \mathbb{R}^{d}\f$
class Domain {
public:
  
  /**
     @param[in] dim The dimension \f$d\f$
   */
  Domain(std::size_t const dim);

  virtual ~Domain() = default;

  /// Is a point inside the domain?
  /**
     @param[in] x The point
     \return <tt>true</tt>: Then \f$x \in \Omega\f$, <tt>false</tt>: Then \f$x \notin \Omega\f$
  */
  virtual bool Inside(Eigen::VectorXd const& x) const;

  /// Map to a hypercube \f$[-1, 1]^d\f$
  /**
     @param[in] x A point in the domain \f$x \in \Omega\f$
     \return A point in the hypercube \f$[-1, 1]^d\f$
   */
  virtual Eigen::VectorXd MapToHypercube(Eigen::VectorXd const& x) const;

  /// Generate a sample in the domain
  /**
     \return A point in the domain
   */
  virtual Eigen::VectorXd Sample();

  /// The dimension of the domain \f$d\f$
  const std::size_t dim;

private:
  
};
  
} // namespace clf

#endif
