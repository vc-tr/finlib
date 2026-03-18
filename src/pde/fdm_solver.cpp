#include "qe/pde/fdm_solver.hpp"
#include "qe/math/linalg.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace qe {

Real FDMSolver::payoff(Real S) const {
    if (config_.type == OptionType::Call)
        return std::max(S - config_.strike, 0.0);
    return std::max(config_.strike - S, 0.0);
}

Real FDMSolver::lower_boundary(Real S_min, Real t) const {
    Real tau = config_.maturity - t;  // time remaining
    if (config_.type == OptionType::Call) {
        // Call at S → 0: value → 0
        return 0.0;
    }
    // Put at S → 0: value → K * exp(-r * tau)
    return config_.strike * std::exp(-config_.rate * tau) - S_min;
}

Real FDMSolver::upper_boundary(Real S_max, Real t) const {
    Real tau = config_.maturity - t;
    if (config_.type == OptionType::Call) {
        // Call at S → ∞: value → S - K * exp(-r * tau)
        return S_max - config_.strike * std::exp(-config_.rate * tau);
    }
    // Put at S → ∞: value → 0
    return 0.0;
}

FDMResult FDMSolver::solve(FDMScheme scheme) const {
    switch (scheme) {
        case FDMScheme::Explicit: return solve_explicit();
        case FDMScheme::Implicit: return solve_implicit();
        case FDMScheme::CrankNicolson: return solve_crank_nicolson();
    }
    throw std::runtime_error("Unknown FDM scheme");
}

FDMResult FDMSolver::extract_result(const Vec& S_grid, const Vec& V, Real dt) const {
    // Find the grid point closest to spot
    Size idx = 0;
    Real min_dist = std::abs(S_grid[0] - config_.spot);
    for (Size i = 1; i < S_grid.size(); ++i) {
        Real dist = std::abs(S_grid[i] - config_.spot);
        if (dist < min_dist) {
            min_dist = dist;
            idx = i;
        }
    }

    // Ensure we're not at boundary
    if (idx == 0) idx = 1;
    if (idx >= S_grid.size() - 1) idx = S_grid.size() - 2;

    Real price = V[idx];

    // Delta: dV/dS via central difference
    Real dS = S_grid[idx + 1] - S_grid[idx - 1];
    Real delta = (V[idx + 1] - V[idx - 1]) / dS;

    // Gamma: d2V/dS2
    Real dS1 = S_grid[idx + 1] - S_grid[idx];
    Real dS0 = S_grid[idx] - S_grid[idx - 1];
    Real gamma = 2.0 * (V[idx + 1] / (dS1 * (dS1 + dS0))
                       - V[idx] / (dS1 * dS0)
                       + V[idx - 1] / (dS0 * (dS1 + dS0)));

    // Theta: approximate from BS PDE relation
    // theta = -0.5*sigma^2*S^2*gamma - r*S*delta + r*V
    Real S = S_grid[idx];
    Real theta = -0.5 * config_.sigma * config_.sigma * S * S * gamma
                 - config_.rate * S * delta + config_.rate * price;

    return {price, delta, gamma, theta, S_grid, V};
}

// ============================================================================
// Explicit scheme (forward Euler)
// ============================================================================
FDMResult FDMSolver::solve_explicit() const {
    Size M = config_.n_spot;
    Size N = config_.n_time;

    Real S_min = config_.spot * config_.spot_min_mult;
    Real S_max = config_.spot * config_.spot_max_mult;
    Real dS = (S_max - S_min) / static_cast<Real>(M);
    Real dt = config_.maturity / static_cast<Real>(N);

    // Build spot grid
    Vec S_grid(M + 1);
    for (Size i = 0; i <= M; ++i) {
        S_grid[i] = S_min + static_cast<Real>(i) * dS;
    }

    // Terminal condition
    Vec V(M + 1);
    for (Size i = 0; i <= M; ++i) {
        V[i] = payoff(S_grid[i]);
    }

    // March backward in time
    Real r = config_.rate;
    Real sig2 = config_.sigma * config_.sigma;

    for (Size n = N; n > 0; --n) {
        Real t = static_cast<Real>(n - 1) * dt;
        Vec V_new(M + 1);

        // Boundary conditions
        V_new[0] = lower_boundary(S_grid[0], t);
        V_new[M] = upper_boundary(S_grid[M], t);

        // Interior points
        for (Size i = 1; i < M; ++i) {
            Real S = S_grid[i];
            Real alpha = 0.5 * sig2 * S * S / (dS * dS);
            Real beta = r * S / (2.0 * dS);

            Real a = (alpha - beta) * dt;
            Real b = 1.0 - (2.0 * alpha + r) * dt;
            Real c = (alpha + beta) * dt;

            V_new[i] = a * V[i - 1] + b * V[i] + c * V[i + 1];
        }

        V = V_new;
    }

    return extract_result(S_grid, V, dt);
}

// ============================================================================
// Implicit scheme (backward Euler, Thomas algorithm)
// ============================================================================
FDMResult FDMSolver::solve_implicit() const {
    Size M = config_.n_spot;
    Size N = config_.n_time;

    Real S_min = config_.spot * config_.spot_min_mult;
    Real S_max = config_.spot * config_.spot_max_mult;
    Real dS = (S_max - S_min) / static_cast<Real>(M);
    Real dt = config_.maturity / static_cast<Real>(N);

    Vec S_grid(M + 1);
    for (Size i = 0; i <= M; ++i) {
        S_grid[i] = S_min + static_cast<Real>(i) * dS;
    }

    Vec V(M + 1);
    for (Size i = 0; i <= M; ++i) {
        V[i] = payoff(S_grid[i]);
    }

    Real r = config_.rate;
    Real sig2 = config_.sigma * config_.sigma;

    for (Size n = N; n > 0; --n) {
        Real t = static_cast<Real>(n - 1) * dt;

        // Build tridiagonal system for interior points [1..M-1]
        Size m = M - 1;  // number of interior points
        Vec sub(m - 1), diag(m), sup(m - 1), rhs(m);

        for (Size j = 0; j < m; ++j) {
            Size i = j + 1;  // actual grid index
            Real S = S_grid[i];
            Real alpha = 0.5 * sig2 * S * S / (dS * dS);
            Real beta = r * S / (2.0 * dS);

            Real a = -(alpha - beta) * dt;
            Real b = 1.0 + (2.0 * alpha + r) * dt;
            Real c = -(alpha + beta) * dt;

            diag[j] = b;
            if (j > 0) sub[j - 1] = a;
            if (j < m - 1) sup[j] = c;

            rhs[j] = V[i];
        }

        // Apply boundary conditions
        Real bc_low = lower_boundary(S_grid[0], t);
        Real bc_high = upper_boundary(S_grid[M], t);

        // Adjust RHS for boundary
        Real S1 = S_grid[1];
        Real alpha1 = 0.5 * sig2 * S1 * S1 / (dS * dS);
        Real beta1 = r * S1 / (2.0 * dS);
        rhs[0] -= (-(alpha1 - beta1) * dt) * bc_low;

        Real SM1 = S_grid[M - 1];
        Real alphaM1 = 0.5 * sig2 * SM1 * SM1 / (dS * dS);
        Real betaM1 = r * SM1 / (2.0 * dS);
        rhs[m - 1] -= (-(alphaM1 + betaM1) * dt) * bc_high;

        Vec interior = thomas_solve(sub, diag, sup, rhs);

        V[0] = bc_low;
        for (Size j = 0; j < m; ++j) {
            V[j + 1] = interior[j];
        }
        V[M] = bc_high;
    }

    return extract_result(S_grid, V, dt);
}

// ============================================================================
// Crank-Nicolson scheme (2nd order in time)
// ============================================================================
FDMResult FDMSolver::solve_crank_nicolson() const {
    Size M = config_.n_spot;
    Size N = config_.n_time;

    Real S_min = config_.spot * config_.spot_min_mult;
    Real S_max = config_.spot * config_.spot_max_mult;
    Real dS = (S_max - S_min) / static_cast<Real>(M);
    Real dt = config_.maturity / static_cast<Real>(N);

    Vec S_grid(M + 1);
    for (Size i = 0; i <= M; ++i) {
        S_grid[i] = S_min + static_cast<Real>(i) * dS;
    }

    Vec V(M + 1);
    for (Size i = 0; i <= M; ++i) {
        V[i] = payoff(S_grid[i]);
    }

    Real r = config_.rate;
    Real sig2 = config_.sigma * config_.sigma;

    // Precompute coefficients for each interior grid point
    // a_i, b_i, c_i are the standard FDM coefficients
    Vec ai(M + 1), bi(M + 1), ci(M + 1);
    for (Size i = 1; i < M; ++i) {
        Real S = S_grid[i];
        Real alpha = 0.5 * sig2 * S * S / (dS * dS);
        Real beta = r * S / (2.0 * dS);
        ai[i] = alpha - beta;
        bi[i] = 2.0 * alpha + r;
        ci[i] = alpha + beta;
    }

    for (Size n = N; n > 0; --n) {
        Real t = static_cast<Real>(n - 1) * dt;

        // Boundary conditions at the new time level
        Real bc_low = lower_boundary(S_grid[0], t);
        Real bc_high = upper_boundary(S_grid[M], t);

        Size m = M - 1;  // interior points
        Vec sub(m - 1), diag(m), sup(m - 1), rhs(m);

        for (Size j = 0; j < m; ++j) {
            Size i = j + 1;

            // Crank-Nicolson: 0.5 * (implicit + explicit)
            // LHS: V_new[i] + 0.5*dt*(a_i*V_new[i-1] - b_i*V_new[i] + c_i*V_new[i+1]) * (-1)
            //     = (1 + 0.5*dt*b_i)*V_new[i] - 0.5*dt*a_i*V_new[i-1] - 0.5*dt*c_i*V_new[i+1]
            diag[j] = 1.0 + 0.5 * dt * bi[i];
            if (j > 0) sub[j - 1] = -0.5 * dt * ai[i];
            if (j < m - 1) sup[j] = -0.5 * dt * ci[i];

            // RHS: (1 - 0.5*dt*b_i)*V[i] + 0.5*dt*a_i*V[i-1] + 0.5*dt*c_i*V[i+1]
            rhs[j] = (1.0 - 0.5 * dt * bi[i]) * V[i]
                    + 0.5 * dt * ai[i] * V[i - 1]
                    + 0.5 * dt * ci[i] * V[i + 1];
        }

        // Boundary adjustments:
        // For j=0 (i=1): the implicit sub-diagonal term a_1*V_new[0] moves to RHS
        rhs[0] += 0.5 * dt * ai[1] * bc_low;
        // For j=m-1 (i=M-1): the implicit super-diagonal term c_{M-1}*V_new[M] moves to RHS
        rhs[m - 1] += 0.5 * dt * ci[M - 1] * bc_high;

        Vec interior = thomas_solve(sub, diag, sup, rhs);

        V[0] = bc_low;
        for (Size j = 0; j < m; ++j) {
            V[j + 1] = interior[j];
        }
        V[M] = bc_high;
    }

    return extract_result(S_grid, V, dt);
}

} // namespace qe
