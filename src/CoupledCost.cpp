#include "clf/CoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledCost::CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, pt::ptree const& pt) :
CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients()+neighbor->NumCoefficients())),
point(point),
neighbor(neighbor),
pointBasisEvals(point->EvaluateBasisFunctions(neighbor->x)),
neighborBasisEvals(neighbor->EvaluateBasisFunctions(neighbor->x)),
localNeighborInd(LocalIndex(point, neighbor))
{
  assert(pointBasisEvals.size()==neighborBasisEvals.size());

  // resize the gradient
  this->gradient.resize(inputSizes(0));
}

bool CoupledCost::Coupled() const { return localNeighborInd!=std::numeric_limits<std::size_t>::max(); }

std::size_t CoupledCost::LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor) {
  const std::size_t localInd = point->LocalIndex(neighbor->GlobalIndex());
  return (localInd==0? std::numeric_limits<std::size_t>::max() : localInd);
}

double CoupledCost::Cost(Eigen::VectorXd const& pointCoeffs, Eigen::VectorXd const& neighCoeffs) const {
  if( !Coupled() ) { return 0.0; }

  auto pnt = point.lock();
  auto neigh = neighbor.lock();

  // the difference in the support point output (evaluated at the neighbor point)
  const Eigen::VectorXd diff = pnt->EvaluateLocalFunction(neigh->x, pointCoeffs, pointBasisEvals) - neigh->EvaluateLocalFunction(neigh->x, neighCoeffs, neighborBasisEvals);

  return pnt->couplingScale*pnt->NearestNeighborKernel(localNeighborInd)*diff.dot(diff)/2.0;
}

double CoupledCost::CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) {
  auto pnt = point.lock();
  auto neigh = neighbor.lock();

  // coefficients for the point and the neighbor point
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&input[0] (0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&input[0] (pnt->NumCoefficients()), neigh->NumCoefficients());

  return Cost(pointCoeffs, neighCoeffs);
}

Eigen::VectorXd CoupledCost::Gradient(Eigen::VectorXd const& pointCoeffs, Eigen::VectorXd const& neighCoeffs) const {
  if( !Coupled() ) { return Eigen::VectorXd::Zero(inputSizes(0)); }

  auto pnt = point.lock();
  auto neigh = neighbor.lock();

  // the difference in the support point output (evaluated at the neighbor point)
  const Eigen::VectorXd diff = pnt->EvaluateLocalFunction(neigh->x, pointCoeffs, pointBasisEvals) - neigh->EvaluateLocalFunction(neigh->x, neighCoeffs, neighborBasisEvals);

  // loop through each output
  std::size_t indPoint = 0, indNeigh = pnt->NumCoefficients();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputSizes(0));
  for( std::size_t i=0; i<diff.size(); ++i ) {
    grad.segment(indPoint, pointBasisEvals[i].size()) = diff(i)*pointBasisEvals[i];
    indPoint += pointBasisEvals[i].size();

    grad.segment(indNeigh, neighborBasisEvals[i].size()) = -diff(i)*neighborBasisEvals[i];
    indNeigh += neighborBasisEvals[i].size();
  }

  return pnt->couplingScale*pnt->NearestNeighborKernel(localNeighborInd)*grad;
}

void CoupledCost::GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) {
  if( !Coupled() ) {
    this->gradient = Eigen::VectorXd::Zero(inputSizes(0));
    return;
  }

  auto pnt = point.lock();
  auto neigh = neighbor.lock();

  // coefficients for the point and the neighbor point
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&input[0] (0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&input[0] (pnt->NumCoefficients()), neigh->NumCoefficients());

  this->gradient = Gradient(pointCoeffs, neighCoeffs)*sensitivity(0);
}

void CoupledCost::Hessian(std::vector<Eigen::MatrixXd>& ViVi, std::vector<Eigen::MatrixXd>& ViVj, std::vector<Eigen::MatrixXd>& VjVj) const {
  if( !Coupled() ) {
    ViVi.clear(); ViVj.clear(); VjVj.clear();
    return;
  }

  auto pnt = point.lock();
  auto neigh = neighbor.lock();
  assert(pnt->model->outputDimension==neigh->model->outputDimension);

  // the scaling constant
  const double scale = pnt->couplingScale*pnt->NearestNeighborKernel(localNeighborInd);

  // loop through each output
  ViVi.resize(pnt->model->outputDimension);
  ViVj.resize(pnt->model->outputDimension);
  VjVj.resize(pnt->model->outputDimension);
  for( std::size_t i=0; i<pnt->model->outputDimension; ++i ) {
    ViVi[i] = scale*pointBasisEvals[i]*pointBasisEvals[i].transpose();
    ViVj[i] = -scale*pointBasisEvals[i]*neighborBasisEvals[i].transpose();
    VjVj[i] = scale*neighborBasisEvals[i]*neighborBasisEvals[i].transpose();
  }
}
