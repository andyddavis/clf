#include "clf/FeatureMatrix.hpp"

using namespace clf;

FeatureMatrix::FeatureMatrix(std::shared_ptr<const FeatureVector> const& vec, std::size_t const numFeatureVectors, std::shared_ptr<Domain> const& domain) :
  numBasisFunctions(numFeatureVectors*vec->NumBasisFunctions()),
  numFeatureVectors(numFeatureVectors),
  featureVectors(std::vector<VectorPair>(1, VectorPair(vec, numFeatureVectors))),
  domain(domain)
{}

FeatureMatrix::FeatureMatrix(std::vector<VectorPair> const& featureVectors, std::shared_ptr<Domain> const& domain) :
  numBasisFunctions(ComputeNumBasisFunctions(featureVectors)),
  numFeatureVectors(ComputeNumFeatureVectors(featureVectors)),
  featureVectors(featureVectors),
  domain(domain)
{
  // make sure the feature vectors have the same input dimension
  assert(featureVectors.size()>0);
  for( auto it=featureVectors.begin()+1; it!=featureVectors.end(); ++it ) { assert(featureVectors[0].first->InputDimension()==it->first->InputDimension()); }
}

FeatureMatrix::FeatureMatrix(std::vector<std::shared_ptr<const FeatureVector> > const& vecs, std::shared_ptr<Domain> const& domain) :
  numBasisFunctions(ComputeNumBasisFunctions(vecs)),
  numFeatureVectors(vecs.size()),
  featureVectors(CreateVectorPairs(vecs)),
  domain(domain)
{
  // make sure the feature vectors have the same input dimension
  assert(featureVectors.size()>0);
  for( auto it=featureVectors.begin()+1; it!=featureVectors.end(); ++it ) { assert(featureVectors[0].first->InputDimension()==it->first->InputDimension()); }
}

std::vector<FeatureMatrix::VectorPair> FeatureMatrix::CreateVectorPairs(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
  std::vector<VectorPair> pairs(vecs.size());
  for( std::size_t i=0; i<vecs.size(); ++i ) { pairs[i] = VectorPair(vecs[i], 1); }
  return pairs;
}

std::size_t FeatureMatrix::ComputeNumBasisFunctions(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
  std::size_t num = 0;
  for( const auto& it : vecs ) { num += it->NumBasisFunctions(); }
  return num; 
}

std::size_t FeatureMatrix::ComputeNumBasisFunctions(std::vector<VectorPair> const& featureVectors) { 
  std::size_t num = 0;
  for( const auto& it : featureVectors ) { num += it.second*it.first->NumBasisFunctions(); }
  return num; 
}

std::size_t FeatureMatrix::ComputeNumFeatureVectors(std::vector<VectorPair> const& featureVectors) { 
  std::size_t num = 0;
  for( const auto& it : featureVectors ) { num += it.second; }
  return num; 
}

std::size_t FeatureMatrix::InputDimension() const { return featureVectors[0].first->InputDimension(); }

Eigen::VectorXd FeatureMatrix::ApplyTranspose(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  assert(coeff.size()==numBasisFunctions);
  Eigen::VectorXd output(numFeatureVectors);

  std::size_t count = 0, start = 0;
  const std::optional<Eigen::VectorXd> y = LocalCoordinate(x);
  for( const auto& it : featureVectors ) {
    const Eigen::VectorXd phi = it.first->Evaluate((y? *y : x));
    for( std::size_t i=0; i<it.second; ++i ) {
      output(count) = phi.dot(coeff.segment(start, phi.size()));

      ++count;
      start += it.first->NumBasisFunctions();
    }
  }
  assert(count==numFeatureVectors);
  assert(start==numBasisFunctions);

  return output;
}

Eigen::MatrixXd FeatureMatrix::ApplyTranspose(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, std::shared_ptr<LinearDifferentialOperator> const& linOper) const {
  assert(coeff.size()==numBasisFunctions);
  Eigen::MatrixXd output(numFeatureVectors, linOper->NumOperators());

  // the jacobian of the coordinate transformation
  const std::optional<Eigen::VectorXd> y = LocalCoordinate(x);
  const std::optional<Eigen::VectorXd> jac = LocalCoordinateJacobian();

  LinearDifferentialOperator::CountPair oper = linOper->Counts(0);
  std::size_t count = 0, start = 0, diff = oper.second;
  for( const auto& it : featureVectors ) {
    Eigen::MatrixXd phi = it.first->Derivative((y? *y : x), oper.first, jac).transpose();
    for( std::size_t i=0; i<it.second; ++i ) {
      output.row(count) = phi*coeff.segment(start, it.first->NumBasisFunctions());
       
      ++count;
      start += it.first->NumBasisFunctions();
      if( diff<=count && count!=numFeatureVectors ) {
      	oper = linOper->Counts(count);
	diff += oper.second;
      	phi = it.first->Derivative((y? *y : x), oper.first, jac).transpose();
      }
    }
  }

  return output;
}

std::optional<Eigen::VectorXd> FeatureMatrix::LocalCoordinateJacobian() const {
  if( domain ) { return domain->MapToHypercubeJacobian(); }
  return std::nullopt;
}

std::optional<Eigen::VectorXd> FeatureMatrix::LocalCoordinate(Eigen::VectorXd const& x) const {
  if( domain ) { return domain->MapToHypercube(x); }
  return std::nullopt;
}

std::shared_ptr<const FeatureVector> FeatureMatrix::GetFeatureVector(std::size_t const ind) const {
  assert(ind<numFeatureVectors);

  std::size_t jnd = 0;
  for( auto const& it : featureVectors ) {
    jnd += it.second;
    if( ind<jnd ) { return it.first; }
  }
  return nullptr;
}

std::vector<FeatureMatrix::VectorPair>::const_iterator FeatureMatrix::Begin() const { return featureVectors.begin(); }

std::vector<FeatureMatrix::VectorPair>::const_iterator FeatureMatrix::End() const { return featureVectors.end(); }
