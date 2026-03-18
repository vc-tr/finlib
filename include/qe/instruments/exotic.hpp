#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/math/linalg.hpp"
#include "qe/models/black_scholes.hpp"
#include "qe/montecarlo/engine.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace qe {

// ============================================================================
// Barrier option types
// ============================================================================
enum class BarrierType {
    DownAndOut,   // Knocked out if S <= barrier
    DownAndIn,    // Activated if S <= barrier
    UpAndOut,     // Knocked out if S >= barrier
    UpAndIn       // Activated if S >= barrier
};

// ============================================================================
// Asian option average type
// ============================================================================
enum class AverageType {
    Arithmetic,
    Geometric
};

// ============================================================================
// Exotic MC Pricer — prices path-dependent options via Monte Carlo
// ============================================================================
class ExoticPricer {
public:
    struct Config {
        Real spot;
        Real rate;
        Real sigma;
        Real maturity;
        Size num_paths = 100000;
        Size num_steps = 252;    // daily steps for path-dependent
        uint64_t seed = 42;
    };

    explicit ExoticPricer(const Config& config) : config_(config) {}

    // ----- Barrier Options -----
    struct BarrierResult {
        Real price;
        Real std_error;
        Real knock_pct;   // fraction of paths that hit the barrier
    };

    BarrierResult price_barrier(Real strike, OptionType opt_type,
                                Real barrier, BarrierType bar_type) const;

    // ----- Asian Options -----
    MCResult price_asian(Real strike, OptionType opt_type,
                         AverageType avg_type) const;

    // Closed-form geometric Asian call (for verification)
    static Real geometric_asian_call(Real S, Real K, Real r,
                                      Real sigma, Real T, Size n_steps);

    // ----- Lookback Options -----
    // Floating strike lookback: call pays S(T) - min(S), put pays max(S) - S(T)
    MCResult price_lookback(OptionType type) const;

    // ----- Basket Options -----
    // Multi-asset option on weighted average of correlated assets
    struct BasketConfig {
        Vec spots;       // initial prices per asset
        Vec sigmas;      // volatilities per asset
        Vec weights;     // portfolio weights (sum to 1)
        Mat correlation; // correlation matrix
    };

    MCResult price_basket(const BasketConfig& basket, Real strike,
                          OptionType type) const;

private:
    Config config_;

    // Generate one GBM path, returns all spot values [0..n_steps]
    Vec generate_path(MersenneTwister& rng) const;
};

} // namespace qe
