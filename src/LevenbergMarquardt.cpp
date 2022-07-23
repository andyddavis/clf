#include "clf/LevenbergMarquardt.hpp"

using namespace clf;

DenseLevenbergMarquardt::DenseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::MatrixXd> > const& cost, std::shared_ptr<Parameters> const& para) :
  LevenbergMarquardt<Eigen::MatrixXd>(cost, para)
{}

void DenseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const { mat += scale*Eigen::MatrixXd::Identity(mat.rows(), mat.cols()); }

SparseLevenbergMarquardt::SparseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::SparseMatrix<double> > > const& cost, std::shared_ptr<Parameters> const& para) :
  LevenbergMarquardt<Eigen::SparseMatrix<double> >(cost, para)
{}

void SparseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const {
  for( std::size_t i=0; i<std::min(mat.rows(), mat.cols()); ++i ) { mat.coeffRef(i, i) += scale; }
  mat.makeCompressed();
}
