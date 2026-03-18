#include "qe/math/interpolation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qe {

// ============================================================================
// LinearInterpolator
// ============================================================================

LinearInterpolator::LinearInterpolator(std::span<const Real> x, std::span<const Real> y)
    : x_(x.begin(), x.end()), y_(y.begin(), y.end()) {
    if (x.size() != y.size()) throw std::invalid_argument("LinearInterpolator: x and y must have same size");
    if (x.size() < 2) throw std::invalid_argument("LinearInterpolator: need at least 2 points");
}

Size LinearInterpolator::find_segment(Real xi) const {
    if (xi <= x_.front()) return 0;
    if (xi >= x_.back()) return x_.size() - 2;

    auto it = std::lower_bound(x_.begin(), x_.end(), xi);
    Size idx = static_cast<Size>(it - x_.begin());
    return (idx == 0) ? 0 : idx - 1;
}

Real LinearInterpolator::operator()(Real xi) const {
    Size i = find_segment(xi);
    Real t = (xi - x_[i]) / (x_[i + 1] - x_[i]);
    return y_[i] + t * (y_[i + 1] - y_[i]);
}

Vec LinearInterpolator::interpolate(std::span<const Real> xi) const {
    Vec result(xi.size());
    for (Size i = 0; i < xi.size(); ++i) {
        result[i] = (*this)(xi[i]);
    }
    return result;
}

// ============================================================================
// CubicSplineInterpolator
// ============================================================================

CubicSplineInterpolator::CubicSplineInterpolator(std::span<const Real> x, std::span<const Real> y)
    : x_(x.begin(), x.end()), y_(y.begin(), y.end()) {
    if (x.size() != y.size()) throw std::invalid_argument("CubicSplineInterpolator: x and y must have same size");
    if (x.size() < 3) throw std::invalid_argument("CubicSplineInterpolator: need at least 3 points");
    compute_coefficients();
}

void CubicSplineInterpolator::compute_coefficients() {
    Size n = x_.size() - 1;  // number of segments

    // Step sizes
    Vec h(n);
    for (Size i = 0; i < n; ++i) {
        h[i] = x_[i + 1] - x_[i];
    }

    // Set up tridiagonal system for natural cubic spline
    // Solve for second derivatives (c coefficients)
    // Natural BC: c[0] = c[n] = 0
    Vec alpha(n);
    for (Size i = 1; i < n; ++i) {
        alpha[i] = (3.0 / h[i]) * (y_[i + 1] - y_[i])
                 - (3.0 / h[i - 1]) * (y_[i] - y_[i - 1]);
    }

    // Thomas algorithm for tridiagonal system
    Vec l(n + 1), mu(n + 1), z(n + 1);
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;

    for (Size i = 1; i < n; ++i) {
        l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n] = 1.0;
    z[n] = 0.0;

    c_.resize(n + 1);
    b_.resize(n);
    d_.resize(n);
    a_.resize(n);

    c_[n] = 0.0;

    for (Size j = n; j > 0; --j) {
        Size i = j - 1;
        c_[i] = z[i] - mu[i] * c_[i + 1];
        b_[i] = (y_[i + 1] - y_[i]) / h[i] - h[i] * (c_[i + 1] + 2.0 * c_[i]) / 3.0;
        d_[i] = (c_[i + 1] - c_[i]) / (3.0 * h[i]);
        a_[i] = y_[i];
    }
}

Size CubicSplineInterpolator::find_segment(Real xi) const {
    if (xi <= x_.front()) return 0;
    if (xi >= x_.back()) return x_.size() - 2;

    auto it = std::lower_bound(x_.begin(), x_.end(), xi);
    Size idx = static_cast<Size>(it - x_.begin());
    return (idx == 0) ? 0 : idx - 1;
}

Real CubicSplineInterpolator::operator()(Real xi) const {
    Size i = find_segment(xi);
    Real dx = xi - x_[i];
    return a_[i] + b_[i] * dx + c_[i] * dx * dx + d_[i] * dx * dx * dx;
}

Real CubicSplineInterpolator::derivative(Real xi) const {
    Size i = find_segment(xi);
    Real dx = xi - x_[i];
    return b_[i] + 2.0 * c_[i] * dx + 3.0 * d_[i] * dx * dx;
}

Vec CubicSplineInterpolator::interpolate(std::span<const Real> xi) const {
    Vec result(xi.size());
    for (Size i = 0; i < xi.size(); ++i) {
        result[i] = (*this)(xi[i]);
    }
    return result;
}

} // namespace qe
