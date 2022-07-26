#include "clf/FiniteDifference.hpp"

Eigen::VectorXd clf::FiniteDifference::Weights(std::size_t const order) {
  Eigen::VectorXd weights;
  switch( order ) { 
  case 2: 
    weights.resize(1);
    weights << 0.5;
    return weights;
  case 4: 
    weights.resize(2);
    weights << 2.0/3.0, -1.0/12.0;
    return weights; 
  case 6: 
    weights.resize(3);
    weights << 0.75, -3.0/20.0, 1.0/60;
    return weights;
  default: // default to 8th order
    weights.resize(4);
    weights << 0.8, -0.2, 4.0/105.0, -1.0/280.0;
    return weights;
  }
}
