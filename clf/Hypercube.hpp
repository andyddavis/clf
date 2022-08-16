#ifndef HYPERCUBE_HPP_
#define HYPERCUBE_HPP_

#include <memory>
#include <random>

#include "clf/Parameters.hpp"

#include "clf/Domain.hpp"

namespace clf {

/// A hypercube domain \f$[\ell_0, r_0] \times [\ell_1, r_1] \times ... \times [\ell_d, r_d]\f$
class Hypercube : public Domain {
public:

  /// Create a unit hyper cube \f$[0, 1]^{d}\f$
  /**
     @param[in] dim The dimension \f$d\f$
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(std::size_t const dim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /// Create a hypercube \f$[\ell, r]^{d}\f$
  /**
     @param[in] left The left boundary \f$\ell\f$
     @param[in] right The right boundary \f$r\f$
     @param[in] dim The dimension \f$d\f$ 
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(double const left, double const right, std::size_t const dim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /// Create a hyper cube \f$[\ell, r]^{d}\f$
  /**
     <B>Additional Configuration Parameters:</B>
     Parameter Key | Type | Default Value | Description |
     ------------- | ------------- | ------------- | ------------- |
     "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension \f$d\f$. This is a required parameter |
     "LeftBoundary"   | <tt>std::optional<double></tt> | <tt>std::nullopt</tt> | The left boundary \f$\ell\f$ for <em>all</em> dimensions |
     "RightBoundary"   | <tt>std::optional<double></tt> | <tt>std::nullopt</tt> | The right boundary \f$r\f$ for <em>all</em> dimensions |
     @param[in] para Parameters for this hypercube
   */
  Hypercube(std::shared_ptr<const Parameters> const& para);

  /// Create a hypercube \f$[\ell_0, r_0] \times [\ell_1, r_1] \times ... \times [\ell_d, r_d]\f$
  /**
     @param[in] left The left boundaries \f$\ell_i\f$
     @param[in] right The right boundaries \f$r_i\f$
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~Hypercube() = default;

  /// The left boundary of the \f$i^{\text{th}}\f$ dimension \f$\ell_i\f$
  /**
     @param[in] ind The index \f$i\f$
     \return The left boundary of the \f$i^{\text{th}}\f$ dimension
   */
  double LeftBoundary(std::size_t const ind) const;

  /// The right boundary of the \f$i^{\text{th}}\f$ dimension \f$r_i\f$
  /**
     @param[in] ind The index \f$i\f$
     \return The right boundary of the \f$i^{\text{th}}\f$ dimension
   */
  double RightBoundary(std::size_t const ind) const;

protected:
  
  /// Is a point inside the domain?
  /**
     @param[in] x The point
     \return <tt>true</tt>: Then \f$x \in \Omega\f$, <tt>false</tt>: Then \f$x \notin \Omega\f$
  */
  virtual bool CheckInside(Eigen::VectorXd const& x) const final override;
  
  /// Generate a sample in the domain
  /**
     \return A point in the domain
   */
  virtual Eigen::VectorXd ProposeSample() final override;

private:

  /// Create a random number generator
  /**
     \return The random number generator
   */
  static std::mt19937_64 RandomNumberGenerator();

  void ComputeMapToHypercube();

  /// The random number generator
  static std::mt19937_64 gen;

  /// The uniform distribution(s) that allows use generate random points in this domain
  std::vector<std::uniform_real_distribution<double> > sampler;
};
  
} // namespace clf

#endif
