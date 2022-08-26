#ifndef SPARSELEVENBERGMARQUARDT_HPP_
#define SPARSELEVENBERGMARQUARDT_HPP_

#include "clf/SparseCostFunction.hpp"
#include "clf/LevenbergMarquardt.hpp"

namespace clf {
  
/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using sparse matrices
class SparseLevenbergMarquardt : public LevenbergMarquardt<Eigen::SparseMatrix<double> > {
public:

  /**
     @param[in] cost The cost function that we are trying to minimize
     @param[in] para The parameters for this algorithm
   */
  SparseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::SparseMatrix<double> > > const& cost, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>());

  virtual ~SparseLevenbergMarquardt() = default;

protected:

  /// Add a scaled identity to a matrix
  /**
     Dense and sparse matrices do this slightly differently 
     @param[in] scale Add this number times and identity to the matrix 
     @param[in, out] mat We are adding to this matrix
   */
  virtual void AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const final override;
  
private:
};

} // namespace clf

#endif
