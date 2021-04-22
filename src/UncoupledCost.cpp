#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

using namespace muq::Optimization;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point) : CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients())), point(point) {}

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

  return cost/(2.0*pnt->NumNeighbors());
}

 void UncoupledCost::GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) {
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
   }

   this->gradient = Eigen::VectorXd::Zero(inputSizes(0));
   this->gradient *= sensitivity(0);
 }
