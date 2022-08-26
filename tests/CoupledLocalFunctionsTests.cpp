#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/IdentityModel.hpp"
#include "clf/CoupledLocalFunctions.hpp"
#include "clf/BoundaryCondition.hpp"
#include "clf/AdvectionEquation.hpp"

using namespace clf;

TEST(CoupledLocalFunctionsTests, EvaluationTests) {
  const std::size_t indim = 5;
  const std::size_t outdim = 1;
  const std::size_t numPoints = 100;
  const std::size_t numLocalPoints = 100;
  const std::size_t maxOrder = 4;

  // parameters
  auto para = std::make_shared<Parameters>();
  para->Add("NumSupportPoints", numPoints);
  para->Add("NumLocalPoints", numLocalPoints);
  para->Add("InputDimension", indim);
  para->Add("OutputDimension", outdim);

  // create the domain
  std::vector<bool> periodic(indim, true);
  periodic[0] = false;
  auto dom = std::make_shared<Hypercube>(periodic);

  // the basis function information
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  auto leg = std::make_shared<LegendrePolynomials>();

  const Eigen::VectorXd delta = 0.1*Eigen::VectorXd::Random(indim).array().abs();
  
  CoupledLocalFunctions func(set, leg, dom, delta, para);
  EXPECT_EQ(func.NumLocalFunctions(), numPoints);
  EXPECT_EQ(func.NumCoefficients(), numPoints*set->NumIndices());

  // create two systems 
  auto system0 = std::make_shared<IdentityModel>(indim, outdim);
  auto system1 = std::make_shared<IdentityModel>(indim, outdim);
  EXPECT_TRUE(system0->id<system1->id);

  // add them in a different order than they were created
  const std::size_t numBoundaryPoints = 500;
  func.AddBoundaryCondition(system1, [indim](std::pair<Eigen::VectorXd, Eigen::VectorXd> const& samp) { return samp.second(0)<0.0 && samp.second.tail(indim-1).norm()<1.0e-15; }, numBoundaryPoints);
  func.AddBoundaryCondition(system0, [indim](std::pair<Eigen::VectorXd, Eigen::VectorXd> const& samp) { return samp.second(0)<0.0 && samp.second.tail(indim-1).norm()<1.0e-15; }, numBoundaryPoints);

  for( std::size_t i=0; i<func.NumLocalFunctions(); ++i ) {
    // get the boundary conditions associated with this function
    std::optional<CoupledLocalFunctions::Residuals> resids = func.GetResiduals(i);
    if( resids ) {   
      // the boundary conditions should be sorted by ID number
      for( const auto& it : *resids ) {
	EXPECT_TRUE((*(resids->begin()))->SystemID()<=it->SystemID());
	auto bc = std::dynamic_pointer_cast<BoundaryCondition>(it);
	EXPECT_TRUE(bc);
      }
    }
  }

  func.RemoveResidual(system0->id);
  for( std::size_t i=0; i<func.NumLocalFunctions(); ++i ) {
    // get the boundary conditions associated with this function
    std::optional<CoupledLocalFunctions::Residuals> resids = func.GetResiduals(i);
    if( resids ) {
      EXPECT_EQ(resids->size(), 1);
      EXPECT_EQ(resids->at(0)->SystemID(), system1->id);
      auto bc = std::dynamic_pointer_cast<BoundaryCondition>(resids->at(0));
      EXPECT_TRUE(bc);
    }
  }

  auto advec = std::make_shared<AdvectionEquation>(indim);
  func.AddResidual(advec, para);
  for( std::size_t i=0; i<func.NumLocalFunctions(); ++i ) {
    // get the boundary conditions associated with this function
    std::optional<CoupledLocalFunctions::Residuals> resids = func.GetResiduals(i);
    EXPECT_TRUE(resids);
    if( resids ) {   
      // the boundary conditions should be sorted by ID number
      for( const auto& it : *resids ) {
	EXPECT_TRUE((*(resids->begin()))->SystemID()<=it->SystemID());
	auto bc = std::dynamic_pointer_cast<BoundaryCondition>(it);
	if( bc ) {
	  EXPECT_EQ(bc->SystemID(), system1->id);
	} else {
	  EXPECT_EQ(it->SystemID(), advec->id);
	}
      }
    }
  }

  func.AddResidual(advec, para);
  for( std::size_t i=0; i<func.NumLocalFunctions(); ++i ) {
    // get the boundary conditions associated with this function
    std::optional<CoupledLocalFunctions::Residuals> resids = func.GetResiduals(i);
    EXPECT_TRUE(resids);
    if( resids ) {   
      // the boundary conditions should be sorted by ID number
      for( auto it=resids->begin(); it!=resids->end(); ++it ) {
	if( it!=resids->begin() ) {
	  EXPECT_TRUE((*(it-1))->SystemID()<(*it)->SystemID());
	} else {
	  EXPECT_EQ((*(resids->begin()))->SystemID(), (*it)->SystemID());
	}
	auto bc = std::dynamic_pointer_cast<BoundaryCondition>(*it);
	if( bc ) {
	  EXPECT_EQ(bc->SystemID(), system1->id);
	} else {
	  EXPECT_EQ((*it)->SystemID(), advec->id);
	}
      }
    }
  }

  func.MinimizeResiduals();
}
