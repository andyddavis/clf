#include "clf/LegendrePolynomials.hpp"

using namespace clf;

LegendrePolynomials::LegendrePolynomials() : 
  OrthogonalPolynomials() {}

double LegendrePolynomials::Phi0() const { return 1.0; }

double LegendrePolynomials::Phi1(double const x) const { return x; }

double LegendrePolynomials::dPhi1dx() const { return 1.0; }

double LegendrePolynomials::Ap(std::size_t const p) const { return (2.0*p-1.0)/(double)p; }

double LegendrePolynomials::Bp(std::size_t const p) const { return 0.0; }

double LegendrePolynomials::Cp(std::size_t const p) const { return (p-1.0)/(double)p; }

