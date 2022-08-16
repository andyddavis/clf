#ifndef TESTDOMAINS_HPP_
#define TESTDOMAINS_HPP_

#include "clf/Domain.hpp"

namespace clf {
namespace tests {

/// A class to test clf::Domain::MapToHypercube 
/**
   Note that this domain is not a valid implementation, we just implement a random map for the function as clf::Domain::MapToHypercube to test the infastructure.
*/
class TestHypercubeMapDomain : public Domain {
public:
  
  /**
     @param[in] dim The dimension \f$d\f$
     @param[in] para The parameters for this domain
  */
  inline TestHypercubeMapDomain(std::size_t const dim) : Domain(dim) {
    map = std::pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::VectorXd::Random(dim), Eigen::VectorXd::Random(dim));
  }

  virtual ~TestHypercubeMapDomain() = default;
  
private:
};
  
} // namespace tests
} // namespace clf

#endif
