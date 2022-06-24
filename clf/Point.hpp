#ifndef POINT_HPP_
#define POINT_HPP_

#include <Eigen/Eigen>

namespace clf {

class Point { 
public:

  Point(Eigen::VectorXd const& x);

  virtual ~Point() = default;

  /// The point location 
  const Eigen::VectorXd x;

private:
};

} // namespace clf 

#endif
