#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include <chrono>

namespace qe {

// Longstaff-Schwartz Least-Squares Monte Carlo for American options
// Uses polynomial regression at each exercise date to estimate
// the continuation value, then decides exercise vs. hold.
class AmericanPricer {
public:
    struct Config {
        Real spot;
        Real strike;
        Real rate;
        Real sigma;
        Real maturity;
        OptionType type;
        Size num_paths = 100000;
        Size num_steps = 50;     // exercise opportunities
        Size poly_degree = 3;    // Laguerre polynomial basis degree
        uint64_t seed = 42;
    };

    struct Result {
        Real price;
        Real std_error;
        Real early_exercise_premium;  // American - European
        Real european_price;          // for comparison
        double elapsed_ms;
    };

    explicit AmericanPricer(const Config& config) : config_(config) {}

    Result price() const;

private:
    Config config_;

    // Generate all paths: returns (num_paths x num_steps+1) matrix
    Mat generate_paths(MersenneTwister& rng) const;

    // Evaluate Laguerre polynomial basis at x
    // L_0(x) = 1, L_1(x) = 1-x, L_2(x) = 1 - 2x + x^2/2, ...
    Vec laguerre_basis(Real x, Size degree) const;

    // Least-squares regression: fit Y = X * beta, return beta
    Vec least_squares(const Mat& X, const Vec& Y) const;

    // Intrinsic value (exercise payoff)
    Real intrinsic(Real S) const;
};

} // namespace qe
