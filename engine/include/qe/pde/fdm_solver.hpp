#pragma once

#include "qe/core/types.hpp"
#include <functional>

namespace qe {

// Finite Difference Method solver for the Black-Scholes PDE
// Backward in time: V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V = 0
//
// Transforms to log-space x = ln(S) for uniform grid:
// V_t + 0.5*sigma^2*V_xx + (r - 0.5*sigma^2)*V_x - r*V = 0

enum class FDMScheme {
    Explicit,       // Forward Euler in time (conditionally stable)
    Implicit,       // Backward Euler in time (unconditionally stable)
    CrankNicolson   // Average of Explicit and Implicit (2nd order in time)
};

struct FDMConfig {
    Real spot;
    Real strike;
    Real rate;
    Real sigma;
    Real maturity;
    OptionType type;

    Size n_spot = 200;     // number of spot grid points
    Size n_time = 1000;    // number of time steps
    Real spot_min_mult = 0.2;   // S_min = spot * mult
    Real spot_max_mult = 3.0;   // S_max = spot * mult
};

struct FDMResult {
    Real price;
    Real delta;
    Real gamma;
    Real theta;
    Vec spot_grid;
    Vec option_values;  // option prices across spot grid at t=0
};

// Solve Black-Scholes PDE using finite difference methods
class FDMSolver {
public:
    explicit FDMSolver(const FDMConfig& config) : config_(config) {}

    FDMResult solve(FDMScheme scheme) const;

private:
    FDMConfig config_;

    // Terminal condition (payoff at maturity)
    Real payoff(Real S) const;

    // Boundary conditions
    Real lower_boundary(Real S_min, Real t) const;
    Real upper_boundary(Real S_max, Real t) const;

    // Solve with explicit scheme
    FDMResult solve_explicit() const;

    // Solve with implicit scheme (Thomas algorithm)
    FDMResult solve_implicit() const;

    // Solve with Crank-Nicolson scheme
    FDMResult solve_crank_nicolson() const;

    // Extract price and Greeks at the spot point
    FDMResult extract_result(const Vec& S_grid, const Vec& V, Real dt) const;
};

} // namespace qe
