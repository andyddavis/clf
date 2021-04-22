#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace muq::Optimization;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point, pt::ptree const& pt) : CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients())), point(point), regularizationScale(pt.get<double>("RegularizationParameter", 0.0)) {}

double UncoupledCost::CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // get the kernel evaluation
  const Eigen::VectorXd kernel = pnt->NearestNeighborKernel();
  assert(kernel.size()==pnt->NumNeighbors());

  // loop through the neighbors
  double cost = 0.0;
  for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) {
    // the location of the neighbor
    const Eigen::VectorXd& neighx = pnt->NearestNeighbor(i);

    // evaluate the difference between model operator and the right hand side
    const Eigen::VectorXd diff = pnt->Operator(neighx, input[0]) - pnt->model->RightHandSide(neighx);

    // add to the cost
    cost += kernel(i)*diff.dot(diff);
  }

  return (cost + (regularizationScale>0.0? regularizationScale*input[0].get().dot(input[0].get()) : 0.0))/(2.0*pnt->NumNeighbors());
}

 void UncoupledCost::GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) {
   // get the support point
   auto pnt = point.lock();
   assert(pnt);

   // get the kernel evaluation
   const Eigen::VectorXd kernel = pnt->NearestNeighborKernel();
   assert(kernel.size()==pnt->NumNeighbors());

   // loop through the neighbors
   this->gradient = Eigen::VectorXd::Zero(inputSizes(0));
   for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) {
     // the location of the neighbor
     const Eigen::VectorXd& neighx = pnt->NearestNeighbor(i);

     // the Jacobian of the operator
     const Eigen::MatrixXd modelJac = pnt->OperatorJacobian(neighx, input[0]);
     assert(modelJac.rows()==pnt->model->outputDimension);
     assert(modelJac.cols()==inputSizes(0));

     this->gradient += kernel(i)*modelJac.transpose()*(pnt->Operator(neighx, input[0]) - pnt->model->RightHandSide(neighx));
   }

   if( regularizationScale>0.0 ) { this->gradient += regularizationScale*input[0]; }
   this->gradient *= sensitivity(0)/pnt->NumNeighbors();
 }

 Eigen::MatrixXd UncoupledCost::Hessian(Eigen::VectorXd const& coefficients, bool const gaussNewtonHessian) const {
   assert(coefficients.size()==inputSizes(0));

   // get the support point
   auto pnt = point.lock();
   assert(pnt);

   // get the kernel evaluation
   const Eigen::VectorXd kernel = pnt->NearestNeighborKernel();
   assert(kernel.size()==pnt->NumNeighbors());

   // loop through the neighbors
   Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(inputSizes(0), inputSizes(0));
   for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) {
     // the location of the neighbor
     const Eigen::VectorXd& neighx = pnt->NearestNeighbor(i);

     // the Jacobian of the operator
     const Eigen::MatrixXd modelJac = pnt->OperatorJacobian(neighx, coefficients);
     assert(modelJac.rows()==pnt->model->outputDimension);
     assert(modelJac.cols()==inputSizes(0));

     hess += modelJac.transpose()*modelJac;

     if( !gaussNewtonHessian ) {
       const std::vector<Eigen::MatrixXd> modelHess = pnt->OperatorHessian(neighx, coefficients);
       assert(modelHess.size()==pnt->model->outputDimension);
       const Eigen::VectorXd diff = pnt->Operator(neighx, coefficients) - pnt->model->RightHandSide(neighx);
       assert(diff.size()==pnt->model->outputDimension);
       for( std::size_t j=0; j<diff.size(); ++j ) {
         assert(modelHess[j].rows()==inputSizes(0));
         assert(modelHess[j].cols()==inputSizes(0));
         hess += modelHess[j]*diff(j);
       }
     }
   }

   if( regularizationScale>0.0 ) { hess += regularizationScale*Eigen::MatrixXd::Identity(inputSizes(0), inputSizes(0)); }
   return hess/pnt->NumNeighbors();
 }
