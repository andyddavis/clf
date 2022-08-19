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

  /// Create a hypercube \f$[\ell, r]^{d}\f$ with (some) periodic boundaries
  /**
     @param[in] periodic Which boundaries are periodic?
     @param[in] left The left boundary \f$\ell\f$
     @param[in] right The right boundary \f$r\f$
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(std::vector<bool> const& periodic, double const left, double const right, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /// Create a hypercube \f$[\ell, r]^{d}\f$
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

  /// Create a hypercube \f$[\ell, r]^{d}\f$ with (some) periodic boundaries
  /**
     <B>Additional Configuration Parameters:</B>
     Parameter Key | Type | Default Value | Description |
     ------------- | ------------- | ------------- | ------------- |
     "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension \f$d\f$. This is a required parameter |
     "LeftBoundary"   | <tt>std::optional<double></tt> | <tt>std::nullopt</tt> | The left boundary \f$\ell\f$ for <em>all</em> dimensions |
     "RightBoundary"   | <tt>std::optional<double></tt> | <tt>std::nullopt</tt> | The right boundary \f$r\f$ for <em>all</em> dimensions |
     @param[in] para Parameters for this hypercube
     @param[in] periodic Which boundaries are periodic?
   */
  Hypercube(std::vector<bool> const& periodic, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /// Create a hypercube \f$[\ell_0, r_0] \times [\ell_1, r_1] \times ... \times [\ell_d, r_d]\f$
  /**
     @param[in] left The left boundaries \f$\ell_i\f$
     @param[in] right The right boundaries \f$r_i\f$
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /// Create a hypercube \f$[\ell_0, r_0] \times [\ell_1, r_1] \times ... \times [\ell_d, r_d]\f$
  /**
     @param[in] left The left boundaries \f$\ell_i\f$
     @param[in] right The right boundaries \f$r_i\f$
     @param[in] periodic Which boundaries are periodic?
     @param[in] para The parameters for this hypercube and its parameter clf::Domain
   */
  Hypercube(Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::vector<bool> const& periodic, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

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

  /// Compute the distance between two points in the domain
  /**
     - If the domain has a super-set then use the super-set distance by default. 
     - If the domain is periodic, use the 2-norm distance in a periodic domain.
     - Otherwise, default to the 2-norm. 
     @param[in] x1 The first point
     @param[in] x2 The second point
     \return The distance between the two points
   */
  virtual double Distance(Eigen::VectorXd const& x1, Eigen::VectorXd const& x2) const final override;

  /// Map into the periodic domain
  /**
     If a coordinate is outside of the \f$[\ell_i, r_i]\f$ bounds but that coordinate is periodic, then map it back into the domain. If no coordinate in the domain is not periodic, return <tt>std::nullopt</tt>
     @param[in] x A point in \f$\mathbb{R}^{d}\f$
     \return A point where periodic coordinates have been modified to be in \f$[\ell_i, r_i]\f$ (if valid)
   */
  std::optional<Eigen::VectorXd> MapPeriodic(Eigen::VectorXd const& x) const;

  /// Is a coordinate periodic
  /**
     @param[in] ind The coordinate
     \return <tt>true</tt>: This is a periodic coordinate, <tt>false</tt>: This is not a periodic coordinate
   */
  bool Periodic(std::size_t const ind) const;
  
  /// Is any coordinate periodic
  /**
     \return <tt>true</tt>: There is a periodic coordinate, <tt>false</tt>: There is not a periodic coordinate
   */
  bool Periodic() const;

  /// Diagonal, linear map to a subset of the hypercube \f$[-1, 1]^d\f$
  /**
     This is defined by a diagonal matrix stored in clf::Domain::diagonalMap. If this is <tt>std::nullopt</tt> then this map is the identity.

     If the domain is periodic or is contained in a periodic domain, then we have an additional shift in this map.
     @param[in] x A point in the domain \f$x \in \Omega\f$
     \return A point in the hypercube \f$[-1, 1]^d\f$
   */
  virtual Eigen::VectorXd MapToHypercube(Eigen::VectorXd const& x) const final override;

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

  /// Generate a sample on the domain boundary
  /**
     \return A point on the domain boundary
   */
  virtual Eigen::VectorXd ProposeBoundarySample() final override;

private:

  /// Create a random number generator
  /**
     \return The random number generator
   */
  static std::mt19937_64 RandomNumberGenerator();

  /// Compute the linear map into the \f$[-1,1]^{d}\f$ hypercube
  void ComputeMapToHypercube();

  /// Map a coordinate into the periodic domain
  /**
     If a coordinate is outside of the \f$[\ell_i, r_i]\f$ bounds but that coordinate is periodic, then map it back into the domain. 
     @param[in] ind The coordinate index 
     @param[in] x The coordinate value
     \return A point where periodic coordinates have been modified to be in \f$[\ell_i, r_i]\f$ (if valid)
   */
  double MapPeriodicCoordinate(std::size_t const ind, double x) const;

  /// The random number generator
  static std::mt19937_64 gen;

  /// The uniform distribution(s) that allows use generate random points in this domain
  std::vector<std::uniform_real_distribution<double> > sampler;

  /// Is each coordinate direction periodic?
  /**
     If <tt>std::nullopt</tt>, then no boundary is periodic.
   */
  std::optional<std::vector<bool> > periodic;

  /// Are any of the boundaries periodic?
  const bool hasPeriodicBoundary = false;

  /// Are any of the boundaries not periodic?
  const bool hasNonPeriodicBoundary = true;
};
  
} // namespace clf

#endif
