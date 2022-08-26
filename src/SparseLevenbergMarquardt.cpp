#include "clf/SparseLevenbergMarquardt.hpp"

using namespace clf;

SparseLevenbergMarquardt::SparseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::SparseMatrix<double> > > const& cost, std::shared_ptr<Parameters> const& para) :
  LevenbergMarquardt<Eigen::SparseMatrix<double> >(cost, para)
{}

void SparseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const {
  for( std::size_t i=0; i<std::min(mat.rows(), mat.cols()); ++i ) { mat.coeffRef(i, i) += scale; }
  mat.makeCompressed();
}
