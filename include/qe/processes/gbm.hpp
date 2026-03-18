#pragma once

#include "qe/processes/sde.hpp"
#include <cmath>

namespace qe {

// Geometric Brownian Motion: dS = mu * S * dt + sigma * S * dW
// Exact solution: S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
class GBM : public SDE {
public:
    GBM(Real mu, Real sigma) : mu_(mu), sigma_(sigma) {}

    Real drift(Real /*t*/, Real x) const override {
        return mu_ * x;
    }

    Real diffusion(Real /*t*/, Real x) const override {
        return sigma_ * x;
    }

    Real diffusion_deriv(Real /*t*/, Real /*x*/) const override {
        return sigma_;  // d(sigma*x)/dx = sigma
    }

    // Exact log-normal evolution (no discretization error)
    Real evolve(Real /*t*/, Real x, Real dt, Real dw) const override {
        return x * std::exp((mu_ - 0.5 * sigma_ * sigma_) * dt + sigma_ * std::sqrt(dt) * dw);
    }

    Real mu() const { return mu_; }
    Real sigma() const { return sigma_; }

private:
    Real mu_;
    Real sigma_;
};

} // namespace qe
