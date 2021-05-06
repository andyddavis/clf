#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

GlobalCost::GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
CostFunction(Eigen::VectorXi::Constant(1, cloud->numCoefficients)),
dofIndices(cloud)
{}

double GlobalCost::CostImpl(ref_vector<Eigen::VectorXd> const& input) {
  assert(dofIndices.cloud);
  double cost = 0.0;

  // loop through the support points
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&input[0](dofIndices.globalDoFIndices[(*point)->GlobalIndex()]), (*point)->NumCoefficients());

    // evaluate the uncoupled cost function
    cost += (*point)->uncoupledCost->Cost(coeff);

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      // extract the coefficients associated with the neighbor
      Eigen::Map<const Eigen::VectorXd> coeffNeigh(&input[0](dofIndices.globalDoFIndices[neigh->GlobalIndex()]), neigh->NumCoefficients());
      cost += coupled->Cost(coeff, coeffNeigh);
    }
  }

  return cost;
}

Eigen::VectorXd GlobalCost::Gradient(Eigen::VectorXd const& coefficients) const {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputSizes(0));
  assert(grad.size()==coefficients.size());

  // loop through the support points
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    const std::size_t globalDoFStart = dofIndices.globalDoFIndices[(*point)->GlobalIndex()];
    const std::size_t dofLength = (*point)->NumCoefficients();

    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&coefficients(globalDoFStart), dofLength);

    // uncoupled gradient
    grad.segment(globalDoFStart, dofLength) += (*point)->uncoupledCost->Gradient(coeff);

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      const std::size_t globalDoFStartNeigh = dofIndices.globalDoFIndices[neigh->GlobalIndex()];
      const std::size_t dofLengthNeigh = neigh->NumCoefficients();

      // extract the coefficients associated with the neighbor
      Eigen::Map<const Eigen::VectorXd> coeffNeigh(&coefficients(globalDoFStartNeigh), dofLengthNeigh);

      const Eigen::VectorXd coupledGrad = coupled->Gradient(coeff, coeffNeigh);
      grad.segment(globalDoFStart, dofLength) += coupledGrad.head(dofLength);
      grad.segment(globalDoFStartNeigh, dofLengthNeigh) += coupledGrad.tail(dofLengthNeigh);
    }
  }

  return grad;
}

void GlobalCost::GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) {
  this->gradient = sensitivity(0)*Gradient(input[0]);
  /*this->gradient = Eigen::VectorXd::Zero(inputSizes(0));
  assert(this->gradient.size()==input[0].get().size());

  // loop through the support points
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    const std::size_t globalDoFStart = dofIndices.globalDoFIndices[(*point)->GlobalIndex()];
    const std::size_t dofLength = (*point)->NumCoefficients();

    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&input[0](globalDoFStart), dofLength);

    // uncoupled gradient
    this->gradient.segment(globalDoFStart, dofLength) += (*point)->uncoupledCost->Gradient(coeff);

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      const std::size_t globalDoFStartNeigh = dofIndices.globalDoFIndices[neigh->GlobalIndex()];
      const std::size_t dofLengthNeigh = neigh->NumCoefficients();

      // extract the coefficients associated with the neighbor
      Eigen::Map<const Eigen::VectorXd> coeffNeigh(&input[0](globalDoFStartNeigh), dofLengthNeigh);

      const Eigen::VectorXd coupledGrad = coupled->Gradient(coeff, coeffNeigh);
      this->gradient.segment(globalDoFStart, dofLength) += coupledGrad.head(dofLength);
      this->gradient.segment(globalDoFStartNeigh, dofLengthNeigh) += coupledGrad.tail(dofLengthNeigh);
    }
  }

  this->gradient *= sensitivity(0);*/
}

void GlobalCost::Hessian(Eigen::VectorXd const& coefficients, bool const gaussNewtonHessian, Eigen::SparseMatrix<double>& hess) const {

  // loop through the support points
  std::vector<Eigen::Triplet<double> > entries;
  entries.reserve(dofIndices.maxNonZeros);
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    const std::size_t globalDoFStart = dofIndices.globalDoFIndices[(*point)->GlobalIndex()];
    const std::size_t dofLength = (*point)->NumCoefficients();

    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&coefficients(globalDoFStart), dofLength);

    // compute the local hessian
    Eigen::MatrixXd hessDiag = (*point)->uncoupledCost->Hessian(coeff, gaussNewtonHessian);
    assert(hessDiag.rows()==dofLength); assert(hessDiag.cols()==dofLength);

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      const std::size_t globalDoFStartNeigh = dofIndices.globalDoFIndices[neigh->GlobalIndex()];
      const std::size_t dofLengthNeigh = neigh->NumCoefficients();

      const std::vector<std::shared_ptr<const BasisFunctions> >& pointBases = (*point)->GetBasisFunctions();
      const std::vector<std::shared_ptr<const BasisFunctions> >& neighBases = neigh->GetBasisFunctions();
      assert(pointBases.size()==neighBases.size());

      std::vector<Eigen::MatrixXd> ViVi, ViVj, VjVj;
      coupled->Hessian(ViVi, ViVj, VjVj);
      assert(ViVi.size()==pointBases.size());
      assert(ViVi.size()==ViVj.size()); assert(ViVi.size()==VjVj.size());
      std::size_t indPoint = 0, indNeigh = 0;
      for( std::size_t d=0; d<ViVi.size(); ++d ) {
        for( std::size_t i=0; i<ViVi[d].rows(); ++i ) {
          for( std::size_t j=0; j<ViVi[d].cols(); ++j ) { hessDiag(indPoint+i, indPoint+j) += ViVi[d](i, j); }
          for( std::size_t j=0; j<ViVj[d].cols(); ++j ) {
            entries.emplace_back(globalDoFStart+indPoint+i, globalDoFStartNeigh+indNeigh+j, ViVj[d](i, j));
            entries.emplace_back(globalDoFStartNeigh+indNeigh+j, globalDoFStart+indPoint+i, ViVj[d](i, j));
          }
        }
        indPoint += ViVi[d].rows();

        for( std::size_t i=0; i<VjVj[d].rows(); ++i ) {
          for( std::size_t j=0; j<VjVj[d].cols(); ++j ) { entries.emplace_back(globalDoFStartNeigh+indNeigh+i, globalDoFStartNeigh+indNeigh+j, VjVj[d](i, j)); }
        }
        indNeigh += VjVj[d].rows();
      }
    }

    // add the local hessian into the global hessian
    for( std::size_t i=0; i<dofLength; ++i ) {
      for( std::size_t j=0; j<dofLength; ++j ) {
        if( std::abs(hessDiag(i, j))>0.0 ) { entries.emplace_back(globalDoFStart+i, globalDoFStart+j, hessDiag(i, j)); }
      }
    }

  }

  hess.resize(inputSizes(0), inputSizes(0));
  hess.setFromTriplets(entries.begin(), entries.end());
}
