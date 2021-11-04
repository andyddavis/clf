#include "clf/LaplaceOperator.hpp"

using namespace clf;

LaplaceOperator::LaplaceOperator(std::size_t const indim, std::size_t const outdim) :
LinearModel(indim, outdim)
{}

 Eigen::MatrixXd LaplaceOperator::ModelMatrix(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   assert(bases.size()==outputDimension);

   std::size_t dim = 0;
   for( const auto& it : bases ) { dim += it->NumBasisFunctions(); }

   Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(outputDimension, dim);
   std::size_t pos = 0;
   for( std::size_t i=0; i<bases.size(); ++i ) {
     const std::size_t num = bases[i]->NumBasisFunctions();
     mat.block(i, pos, 1, num) = bases[i]->EvaluateBasisFunctionDerivatives(x, 2).colwise().sum();
     pos += num;
   }

   return mat;
 }
