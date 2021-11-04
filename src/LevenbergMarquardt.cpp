#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CLF_REGISTER_OPTIMIZER(LevenbergMarquardt, DenseLevenbergMarquardt, Eigen::MatrixXd)

 DenseLevenbergMarquardt::DenseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::MatrixXd> > const& cost, pt::ptree const& pt) :
 LevenbergMarquardt<Eigen::MatrixXd>(cost, pt)
 {}

void  DenseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const { 
  std::cout << "DIAG: " << mat.diagonal().transpose() << std::endl << std::endl;
  mat += scale*Eigen::MatrixXd::Identity(mat.rows(), mat.cols()); 
  //mat += scale*mat.diagonal().asDiagonal();
}

CLF_REGISTER_OPTIMIZER(LevenbergMarquardt, SparseLevenbergMarquardt, Eigen::SparseMatrix<double>)

SparseLevenbergMarquardt::SparseLevenbergMarquardt(std::shared_ptr<CostFunction<Eigen::SparseMatrix<double> > > const& cost, pt::ptree const& pt) : LevenbergMarquardt<Eigen::SparseMatrix<double> >(cost, pt)
{}

void SparseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::SparseMatrix<double>& mat) const {
  for( std::size_t i=0; i<std::min(mat.rows(), mat.cols()); ++i ) { mat.coeffRef(i, i) += scale; }
  mat.makeCompressed();
}
