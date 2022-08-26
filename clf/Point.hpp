#ifndef POINT_HPP_
#define POINT_HPP_

#include <optional>
#include <atomic>

#include <Eigen/Eigen>

namespace clf {

/// A point in the compact domain domain \f$x \in \Omega\f$
class Point { 
public:

  /**
     @param[in] x The point location \f$x \in \Omega\f$
  */
  Point(Eigen::VectorXd const& x);

  /**
     @param[in] x The point location on the boundary \f$x \in \partial \Omega\f$
     @param[in] normal The outward pointing normal vector
  */
  Point(Eigen::VectorXd const& x, Eigen::VectorXd const& normal);

  /**
     @param[in] x First: The point location on the boundary \f$x \in \partial \Omega\f$, Second: The outward pointing normal vector
  */
  Point(std::pair<Eigen::VectorXd, Eigen::VectorXd> const& x);

  virtual ~Point() = default;

  /// The point location \f$x \in \Omega\f$
  const Eigen::VectorXd x;

  /// The (unique) ID of this point
  const std::size_t id;

  /// The outward point normal vector if this is a boundary point
  std::optional<const Eigen::VectorXd> normal;

private:
  /// The ID of this next constructed point
  static std::atomic<std::size_t> nextID;
};

} // namespace clf 

#endif
