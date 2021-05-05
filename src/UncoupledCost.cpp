#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace muq::Optimization;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point, pt::ptree const& pt) :
CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients())),
point(point),
uncoupledScale(pt.get<double>("UncoupledScale", 1.0)),
regularizationScale(pt.get<double>("RegularizationParameter", 0.0))
{
  assert(uncoupledScale>-1.0e-10); assert(regularizationScale>-1.0e-10);
}

double UncoupledCost::Cost(Eigen::VectorXd const& coefficients) const {
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
    const Eigen::VectorXd diff = pnt->Operator(neighx, coefficients) - pnt->model->RightHandSide(neighx);

    // add to the cost
    cost += kernel(i)*diff.dot(diff);
  }

  return (uncoupledScale*cost + (regularizationScale>0.0? regularizationScale*coefficients.dot(coefficients) : 0.0))/(2.0*pnt->NumNeighbors());
}

double UncoupledCost::CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) { return Cost(input[0].get()); }

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

   this->gradient *= uncoupledScale;
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

     hess += kernel(i)*modelJac.transpose()*modelJac;

     if( !gaussNewtonHessian ) {
       const std::vector<Eigen::MatrixXd> modelHess = pnt->OperatorHessian(neighx, coefficients);
       assert(modelHess.size()==pnt->model->outputDimension);
       const Eigen::VectorXd diff = pnt->Operator(neighx, coefficients) - pnt->model->RightHandSide(neighx);
       assert(diff.size()==pnt->model->outputDimension);
       for( std::size_t j=0; j<diff.size(); ++j ) {
         assert(modelHess[j].rows()==inputSizes(0));
         assert(modelHess[j].cols()==inputSizes(0));
         hess += diff(j)*kernel(i)*modelHess[j];
       }
     }
   }

   hess *= uncoupledScale;
   if( regularizationScale>0.0 ) { hess += regularizationScale*Eigen::MatrixXd::Identity(inputSizes(0), inputSizes(0)); }
   return hess/pnt->NumNeighbors();
 }
