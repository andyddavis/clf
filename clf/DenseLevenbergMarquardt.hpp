#ifndef DENSELEVENBERGMARQUARDT_HPP_
#define DENSELEVENBERGMARQUARDT_HPP_

#include "clf/LevenbergMarquardt.hpp"
#include "clf/DenseCostFunction.hpp"

namespace clf {

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using dense matrices
class DenseLevenbergMarquardt : public LevenbergMarquardt<Eigen::MatrixXd> {
public:

  /**
     @param[in] cost The cost function that we are trying to minimize
     @param[in] para The parameters for this algorithm
  */
  DenseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::MatrixXd> > const& cost, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>());

  virtual ~DenseLevenbergMarquardt() = default;

protected:

  /// Add a scaled identity to a matrix
  /**
     Dense and sparse matrices do this slightly differently 
     @param[in] scale Add this number times and identity to the matrix 
     @param[in, out] mat We are adding to this matrix
   */
  virtual void AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const final override;
  
private:
};

} // namespace clf

#endif
