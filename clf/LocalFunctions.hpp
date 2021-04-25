#ifndef LOCALFUNCTIONS_HPP_
#define LOCALFUNCTIONS_HPP_

#include "clf/SupportPointCloud.hpp"

namespace clf {

/// The local function, which is an approximation \f$\hat{u} \approx u\f$ of a function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$
/**
Let \f$\Omega \subseteq \mathbb{R}^{d}\f$ be the domain with closure \f$\overline{\Omega}\f$. Let \f$\mathcal{H}\f$ be a Hilbert space with inner product \f$\langle \cdot, \cdot \rangle_{\mathcal{H}}\f$ such that if \f$u \in \mathcal{H}\f$ then \f$u: \Omega \mapsto \mathbb{R}^{m}\f$.

A local function is defined by a cloud of support points \f$\{x_{i}\}_{i=1}^{n}\f$ (see clf::SupportPointCloud) such that each support point \f$i\f$ is associated with \f$m\f$ local functions \f$\ell_{i}^{(r)}: \overline{\Omega} \mapsto \mathbb{R}\f$ for \f$r \in [1,m]\f$.

The local function is \f$\hat{u}:\Omega \mapsto \mathbb{R}^{m}\f$ for \f$\Omega \subseteq \mathbb{R}^{d}\f$. This function is defined piece-wise by functions \f$\ell_i^{(r)}\f$ associated with support point \f$i\f$ and output \f$r\f$. For this support point, the \f$r^{th}\f$ output of \f$\hat{u}\f$ is \f$\ell_i^{(r)} = p_i^{(r)} \cdot \phi_i^{(r)}\f$, where \f$\phi_i^{(r)}\f$ is a vector of basis function evaluations (see clf::BasisFunctions) and \f$p_i^{(r)} \in \mathbb{R}^{q_i^{(r)}}\f$ are the coefficients for the \f$r^{th}\f$ output. Let \f$p_i = [p_i^{(1)}, p_i^{(2)}, ..., p_i^{(m)}]^{\top} \in \mathbb{R}^{\bar{q}_i}\f$ be a vector of all the coefficients associated with support point \f$i\f$.

If \f$I(x)\f$ is the index of the nearest support point to a point \f$x \in \Omega\f$, then  The local function evaluation is \f$\hat{u}(x) = [\ell_{I(x)}^{(1)}(x), \ell_{I(x)}^{(2)}(x), ..., \ell_{I(x)}^{(m)}(x)]^{\top}\f$
*/
class LocalFunctions {
public:

  /**
  @param[in] cloud The support point cloud that stores all of the support points
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

  /// Determine cloest support point and the squared distance to that point
  /**
  @param[in] x We want the closest support point to this point
  \return First: The index of the closest support point, Second: The squared distance to the closest support point
  */
  std::pair<std::size_t, double> NearestNeighbor(Eigen::VectorXd const& x) const;


private:

  /// Compute the optimal coefficients for each support point
  void ComputeOptimalCoefficients();

  /// Compute the optimal coefficients for each support point given that their is no coupling
  /**
  This function assumes that each support point <em>independently</em> solves the problem
  \f{equation*}{
  p_i = \mbox{arg min}_{p \in \mathbb{R}^{\bar{q}_i}} J(p) = \sum_{j=1}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_i(\hat{u}(x_{I(i,j)}, p)) - f_i(x_{I(i,j)}) \|^2 {K_i(x_i, x_{I(i,j)})} + \frac{a_i}{2} \|p\|^2,
  \f}
  where \f$\mathcal{L}_i\f$ and \f$f_i\f$ are the model operator and right hand side associated with support point \f$i\f$, \f$K_i\f$ is a compact kernel function, and \f$a_i \geq 0\f$ is a regulatory parameter.
  */
  void IndependentSupportPoints();

  /// The support point cloud that stores all of the support points
  std::shared_ptr<SupportPointCloud> cloud;

  /// The cost after optimizing the coefficients for each ceoffcient
  /**
  Uncoupled case: this is the average cost over all of the support points
  */
  double cost = std::numeric_limits<double>::infinity();
};

} // namespace clf

#endif
