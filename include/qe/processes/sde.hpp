#pragma once

#include "qe/core/types.hpp"

namespace qe {

// Abstract interface for a 1D stochastic differential equation
// dX_t = drift(t, X_t) dt + diffusion(t, X_t) dW_t
class SDE {
public:
    virtual ~SDE() = default;

    // Drift coefficient mu(t, x)
    virtual Real drift(Real t, Real x) const = 0;

    // Diffusion coefficient sigma(t, x)
    virtual Real diffusion(Real t, Real x) const = 0;

    // Exact evolution if available (default: Euler-Maruyama step)
    virtual Real evolve(Real t, Real x, Real dt, Real dw) const {
        return x + drift(t, x) * dt + diffusion(t, x) * std::sqrt(dt) * dw;
    }

    // Diffusion derivative d(sigma)/dx for Milstein scheme
    virtual Real diffusion_deriv(Real t, Real x) const {
        // Default: central finite difference
        Real eps = 1e-6 * std::max(std::abs(x), 1.0);
        return (diffusion(t, x + eps) - diffusion(t, x - eps)) / (2.0 * eps);
    }
};

} // namespace qe
