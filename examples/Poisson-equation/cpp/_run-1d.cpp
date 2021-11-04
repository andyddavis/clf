#include <iostream>

#include <MUQ/Modeling/Distributions/UniformBox.h>

#include <clf/LaplaceOperator.hpp>
#include <clf/SupportPointCloud.hpp>
#include <clf/CollocationPointCloud.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

/// The Poisson equation is \f$\nabla \cdot \nabla u = f\$.
class PoissonEquation : public LaplaceOperator {
public:

  /**
  @param[in] indim The input dimension \f$d\f$
  @param[in] outdim The output dimension \f$m\f$ (defaults to \f$1\f$)
  */
  inline PoissonEquation(std::size_t const indim = 1, std::size_t const outdim = 1) : LaplaceOperator(indim, outdim) {}

  virtual ~PoissonEquation() = default;

private:
};

/// The collocation point sampler that ensures we enforce boundary conditions
class PoissonCollocationPointSampler : public CollocationPointSampler {
public:

  inline PoissonCollocationPointSampler(double const alpha0, const double alpha1) :
  CollocationPointSampler(std::make_shared<UniformBox>(0.0, 1.0)->AsVariable(), std::make_shared<PoissonEquation>()),
  alpha0(alpha0),
  alpha1(alpha1),
  alpha2(1.0-alpha0-alpha1)
  {
    assert(alpha0>0.0); assert(alpha1>0.0);
    assert(alpha0+alpha1<1.0);
  }

  virtual ~PoissonCollocationPointSampler() = default;

  inline virtual std::shared_ptr<CollocationPoint> Sample(std::size_t const ind, std::size_t const num) const override {
    if( ind==0 ) { return std::make_shared<CollocationPoint>(alpha0, Eigen::VectorXd::Zero(1), std::make_shared<LinearModel>(model->inputDimension, model->outputDimension)); }
    if( ind==1 ) { return std::make_shared<CollocationPoint>(alpha1, Eigen::VectorXd::Ones(1), std::make_shared<LinearModel>(model->inputDimension, model->outputDimension)); }

    return std::make_shared<CollocationPoint>(alpha2/(num-2), SampleLocation(), model);
  }
private:

  /// The weight of the boundary at \f$x=0\f$
  const double alpha0;

  /// The weight of the boundary at \f$x=1\f$
  const double alpha1;

  /// The weight of the interior \f$\alpha_2 = 1-\alpha_0-\alpha_1\f$
  const double alpha2;
};

int main(int argc, char **argv) {
  std::size_t order = 2;

  // create the support point sampler
  pt::ptree ptSampler;
  ptSampler.put("SupportPoint.BasisFunctions", "Basis");
  ptSampler.put("SupportPoint.Basis.Type", "TotalOrderPolynomials");
  ptSampler.put("SupportPoint.Basis.Order", order);
  ptSampler.put("OutputDimension", 1);
  auto supportSampler = std::make_shared<SupportPointSampler>(std::make_shared<UniformBox>(0.0, 1.0)->AsVariable(), ptSampler);

  // create a support point cloud
  pt::ptree ptSupportCloud;
  ptSupportCloud.put("NumSupportPoints", 10);
  auto supportCloud = SupportPointCloud::Construct(supportSampler, ptSupportCloud);

  // create the collocation point sampler
  auto collocationSampler = std::make_shared<PoissonCollocationPointSampler>(0.1, 0.1);

  // create the collocation point cloud
  pt::ptree ptCollocationCloud;
  ptCollocationCloud.put("NumCollocationPoints", 100);
  auto collocationCloud = std::make_shared<CollocationPointCloud>(collocationSampler, supportCloud, ptCollocationCloud);

  // write the support and collocation points to file
  const std::string file = "examples/Poisson-equation/cpp/Poisson-1d.h5";
  supportCloud->WriteToFile(file);
  collocationCloud->WriteToFile(file);
}
