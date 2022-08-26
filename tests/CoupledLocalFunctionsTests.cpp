#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/IdentityModel.hpp"
#include "clf/CoupledLocalFunctions.hpp"

using namespace clf;

TEST(CoupledLocalFunctionsTests, EvaluationTests) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t numPoints = 50;
  const std::size_t maxOrder = 4;

  // parameters
  auto para = std::make_shared<Parameters>();
  para->Add("NumSupportPoints", numPoints);
  para->Add("InputDimension", indim);
  para->Add("OutputDimension", outdim);

  // create the domain
  auto dom = std::make_shared<Hypercube>(indim);

  // the basis function information
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  auto leg = std::make_shared<LegendrePolynomials>();

  const Eigen::VectorXd delta = 0.1*Eigen::VectorXd::Random(indim).array().abs();
  
  CoupledLocalFunctions func(set, leg, dom, delta, para);
  EXPECT_EQ(func.NumLocalFunctions(), numPoints);

  // create two systems 
  auto system0 = std::make_shared<IdentityModel>(indim, outdim);
  auto system1 = std::make_shared<IdentityModel>(indim, outdim);
  EXPECT_TRUE(system0->id<system1->id);

  // add them in a different order than they were created
  const std::size_t numBoundaryPoints = 500;
  func.SetBoundaryCondition(system1, [indim](std::pair<Eigen::VectorXd, Eigen::VectorXd> const& samp) { return samp.second(0)<0.0 && samp.second.tail(indim-1).norm()<1.0e-15; }, numBoundaryPoints);
  func.SetBoundaryCondition(system0, [indim](std::pair<Eigen::VectorXd, Eigen::VectorXd> const& samp) { return samp.second(1)<0.0 && std::abs(samp.second(0))<1.0e-15 && samp.second.tail(indim-2).norm()<1.0e-15; }, numBoundaryPoints);

  for( std::size_t i=0; i<func.NumLocalFunctions(); ++i ) {
    // get the boundary conditions associated with this function
    std::optional<CoupledLocalFunctions::BoundaryConditions> bcs = func.GetBCs(i);
    if( bcs ) {
      // the boundary conditions should be sorted by ID number
      for( const auto& it : *bcs ) { EXPECT_TRUE((*(bcs->begin()))->SystemID()<=it->SystemID()); }
    }
  }
}
