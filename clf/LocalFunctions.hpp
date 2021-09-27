#ifndef LOCALFUNCTIONS_HPP_
#define LOCALFUNCTIONS_HPP_

#include "clf/GlobalCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace clf {

/// The local function, which is an approximation \f$\hat{u} \approx u\f$ of a function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$
/**
Let \f$\Omega \subseteq \mathbb{R}^{d}\f$ be the domain. 

A local function is defined by a cloud of support points \f$\{x_{i}\}_{i=1}^{n}\f$ (see clf::SupportPointCloud) such that each support point \f$i\f$ is associated with a local function \f$\ell_{x_i}: \Omega \mapsto \mathbb{R}^{m}\f$.

The approximation \f$\hat{u}:\Omega \mapsto \mathbb{R}^{m}\f$ is defined piece-wise by functions \f$\ell_{x_i}(x) = \Phi_{x_i}(x) p_i\f$ associated with support point \f$x_i\f$ (see clf::SupportPoint), where \f$p_i \in \mathbb{R}^{\bar{q}_i}\f$ is a vector of coefficients associated with support point \f$i\f$.

If \f$I(x)\f$ is the index of the nearest support point to a point \f$x \in \Omega\f$, then we evaluate the approximation as \f$\hat{u}(x) = \Phi_{x_{I(x)}}(x) p_{I(x)}\f$.
*/
class LocalFunctions {
public:

  /**
  @param[in] cloud The support point cloud that stores the support points \f$\{x_i\}_{i=1}^{n}\f$ 
  @param[in] options Options for the local function
  */
  LocalFunctions(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt);

  virtual ~LocalFunctions() = default;

  /// The cost associated with the coefficients
  /**
  The cost is infinite if we have not yet run the optimization
  */
  double CoefficientCost() const;

  /// Evaluate the local function at a point
  /**
  @param[in] x The point where we are evaluating the local function
  \return The local function evaluation at the nearest support point
  */
  Eigen::VectorXd Evaluate(Eigen::VectorXd const& x) const;

  /// Determine which support point is the nearest neighbor to an input point
  /**
  @param[in] x We want the closest support point to this point
  \return The index of the closest support point
  */
  std::size_t NearestNeighborIndex(Eigen::VectorXd const& x) const;

  /// Determine the squared distance to the nearest support point
  /**
  @param[in] x We want the closest support point to this point
  \return The squared distance to the closest support point
  */
  double NearestNeighborDistance(Eigen::VectorXd const& x) const;

  /// Determine the closest support point and the squared distance to that point
  /**
  @param[in] x We want the closest support point to this point
  \return First: The index of the closest support point, Second: The squared distance to the closest support point
  */
  std::pair<std::size_t, double> NearestNeighbor(Eigen::VectorXd const& x) const;

  /// Compute the optimal coefficients for each support point
  /**
  @param[in] options Options for the optimization algorithm 
  \return The cost associated with the optimal cupport points
  */
  double ComputeOptimalCoefficients(boost::property_tree::ptree const& options);

private:

  /// Create the global cost function
  /**
  @param[in] cloud The support point cloud
  @param[in] pt Construction options
  \return A nullptr if the support points are independent, otherwise return the global cost function
  */
  static std::shared_ptr<GlobalCost> ConstructGlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt);

  /// Compute the optimal coefficients for each support point given that there is no coupling
  /**
  This function assumes that each support point <em>independently</em> solves the problem
  \f{equation*}{
  p_i = \mbox{arg min}_{p \in \mathbb{R}^{\bar{q}_i}} J(p) = \sum_{j=1}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_i(\hat{u}(x_{I(i,j)}, p)) - f_i(x_{I(i,j)}) \|^2 {K_i(x_i, x_{I(i,j)})} + \frac{a_i}{2} \|p\|^2,
  \f}
  where \f$\mathcal{L}_i\f$ and \f$f_i\f$ are the model operator and right hand side associated with support point \f$i\f$, \f$K_i\f$ is a compact kernel function, and \f$a_i \geq 0\f$ is a regulatory parameter.
  @param[in] options Options for the optimization algorithm 
  \return The average of the costs assocaited with each support point
  */
  double ComputeIndependentSupportPoints(boost::property_tree::ptree const& options);

  /// Compute the optimal coefficients for each support point given that their is no coupling
  /**
  This function assumes that each support point solves the problem
  \f{equation*}{
  p_i = \mbox{arg min}_{p \in \mathbb{R}^{\bar{q}_i}} J(p) = \sum_{j=1}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_i(\hat{u}(x_{I(i,j)}, p)) - f_i(x_{I(i,j)}) \|^2 {K_i(x_i, x_{I(i,j)})} + \frac{a_i}{2} \|p\|^2,
  \f}
  where \f$\mathcal{L}_i\f$ and \f$f_i\f$ are the model operator and right hand side associated with support point \f$i\f$, \f$K_i\f$ is a compact kernel function, and \f$a_i \geq 0\f$ is a regulatory parameter.
  \return The cost associated with the optimal cupport points
  */
  double ComputeCoupledSupportPoints();

  /// The support point cloud that stores all of the support points
  std::shared_ptr<SupportPointCloud> cloud;

  /// The cost after optimizing the coefficients for each ceoffcient
  /**
  Uncoupled case: this is the average cost over all of the support points
  */
  double cost = std::numeric_limits<double>::infinity();

  /// The global cost function
  /**
  This is the null pointer if the support points are independent
  */
  std::shared_ptr<GlobalCost> globalCost;
};

} // namespace clf

#endif
