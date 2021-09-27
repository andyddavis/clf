#include "clf/NLoptOptimizer.hpp"

using namespace clf;

CLF_REGISTER_OPTIMIZER(NLopt, DenseNLoptOptimizer, Eigen::MatrixXd)
CLF_REGISTER_OPTIMIZER(NLopt, SparseNLoptOptimizer, Eigen::SparseMatrix<double>)
