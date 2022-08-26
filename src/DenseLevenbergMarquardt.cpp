#include "clf/DenseLevenbergMarquardt.hpp"

using namespace clf;

DenseLevenbergMarquardt::DenseLevenbergMarquardt(std::shared_ptr<const CostFunction<Eigen::MatrixXd> > const& cost, std::shared_ptr<Parameters> const& para) :
  LevenbergMarquardt<Eigen::MatrixXd>(cost, para)
{}

void DenseLevenbergMarquardt::AddScaledIdentity(double const scale, Eigen::MatrixXd& mat) const { mat += scale*Eigen::MatrixXd::Identity(mat.rows(), mat.cols()); }
