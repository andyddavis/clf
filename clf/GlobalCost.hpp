#ifndef GLOBALCOST_HPP_
#define GLOBALCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Optimization/CostFunction.h>

#include "clf/DOFIndices.hpp"

namespace clf {

/// Compute the global cost of the coefficients for <em>all</em> of the support points in a clf::SupportPointCloud
class GlobalCost : public muq::Optimization::CostFunction {
public:

  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  @param[in] pt The options for the global cost function
  */
  GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt);

  virtual ~GlobalCost() = default;
protected:

  /// Compute the cost function
  /**
  @param[in] input There is only one input: the basis function coefficients for <em>all</em> of the support points
  */
  virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

private:

  /// The map from support point indices to the global degree of freedom indices
  const DOFIndices dofIndices;
};

} // namespace clf

#endif
