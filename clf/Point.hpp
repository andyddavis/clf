#ifndef POINT_HPP_
#define POINT_HPP_

#include <Eigen/Eigen>

namespace clf {

/// A point in the compact domain domain \f$x \in \Omega\f$
class Point { 
public:

  /**
     @param[in] x The point location \f$x \in \Omega\f$
  */
  Point(Eigen::VectorXd const& x);

  virtual ~Point() = default;

  /// The point location \f$x \in \Omega\f$
  const Eigen::VectorXd x;

  /// The (unique) ID of this point
  const std::size_t id;

private:
  /// The ID of this next constructed point
  static std::atomic<std::size_t> nextID;
};

} // namespace clf 

#endif
