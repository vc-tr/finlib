#pragma once

#include "qe/core/types.hpp"
#include <span>

namespace qe {

// Linear interpolation between sorted (x, y) data points
class LinearInterpolator {
public:
    LinearInterpolator(std::span<const Real> x, std::span<const Real> y);

    Real operator()(Real xi) const;

    // Interpolate multiple points
    Vec interpolate(std::span<const Real> xi) const;

private:
    Vec x_, y_;
    Size find_segment(Real xi) const;
};

// Natural cubic spline interpolation
// Solves tridiagonal system for second derivatives, gives C2 continuity
class CubicSplineInterpolator {
public:
    CubicSplineInterpolator(std::span<const Real> x, std::span<const Real> y);

    Real operator()(Real xi) const;

    // First derivative at point xi
    Real derivative(Real xi) const;

    Vec interpolate(std::span<const Real> xi) const;

private:
    Vec x_, y_;
    Vec a_, b_, c_, d_;  // spline coefficients for each segment

    void compute_coefficients();
    Size find_segment(Real xi) const;
};

} // namespace qe
