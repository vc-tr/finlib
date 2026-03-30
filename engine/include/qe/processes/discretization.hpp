#pragma once

#include "qe/core/types.hpp"
#include "qe/processes/sde.hpp"
#include <cmath>

namespace qe {

// Euler-Maruyama discretization scheme (strong order 0.5)
// X_{n+1} = X_n + mu(t, X_n) * dt + sigma(t, X_n) * sqrt(dt) * Z
struct EulerMaruyama {
    static Real step(const SDE& sde, Real t, Real x, Real dt, Real dw) {
        return x + sde.drift(t, x) * dt + sde.diffusion(t, x) * std::sqrt(dt) * dw;
    }
};

// Milstein discretization scheme (strong order 1.0)
// Adds the correction term: + 0.5 * sigma * sigma' * (dW^2 - dt)
// This halves the discretization error compared to Euler-Maruyama
struct Milstein {
    static Real step(const SDE& sde, Real t, Real x, Real dt, Real dw) {
        Real sig = sde.diffusion(t, x);
        Real sig_prime = sde.diffusion_deriv(t, x);
        Real sqrt_dt = std::sqrt(dt);
        Real dW = sqrt_dt * dw;

        return x
            + sde.drift(t, x) * dt
            + sig * dW
            + 0.5 * sig * sig_prime * (dW * dW - dt);
    }
};

// Generate a full path using a given discretization scheme
template<typename Scheme = EulerMaruyama>
Vec generate_path(const SDE& sde, Real x0, Real T, Size n_steps, const Vec& normals) {
    Real dt = T / static_cast<Real>(n_steps);
    Vec path(n_steps + 1);
    path[0] = x0;

    for (Size i = 0; i < n_steps; ++i) {
        Real t = static_cast<Real>(i) * dt;
        path[i + 1] = Scheme::step(sde, t, path[i], dt, normals[i]);
    }

    return path;
}

} // namespace qe
