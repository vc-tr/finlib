#pragma once

#include "qe/core/types.hpp"
#include <functional>

namespace qe {

struct RootResult {
    Real root;
    Size iterations;
    bool converged;
};

// Newton-Raphson method
// f: function, df: derivative, x0: initial guess
// tol: tolerance, max_iter: maximum iterations
RootResult newton_raphson(
    const std::function<Real(Real)>& f,
    const std::function<Real(Real)>& df,
    Real x0,
    Real tol = 1e-12,
    Size max_iter = 100
);

// Brent's method (guaranteed convergence for bracketed roots)
// Requires f(a) and f(b) have opposite signs
RootResult brent(
    const std::function<Real(Real)>& f,
    Real a,
    Real b,
    Real tol = 1e-12,
    Size max_iter = 100
);

// Bisection method (simple, robust, guaranteed convergence)
// Requires f(a) and f(b) have opposite signs
RootResult bisection(
    const std::function<Real(Real)>& f,
    Real a,
    Real b,
    Real tol = 1e-12,
    Size max_iter = 100
);

} // namespace qe
