#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/instruments/payoff.hpp"
#include "qe/processes/gbm.hpp"
#include <chrono>
#include <functional>
#include <memory>

namespace qe {

// Result of a Monte Carlo simulation
struct MCResult {
    Real price;
    Real std_error;
    Real ci_lo;          // 95% confidence interval lower
    Real ci_hi;          // 95% confidence interval upper
    Size num_paths;
    double elapsed_ms;
};

// Variance reduction method tag types
struct NoReduction {};
struct Antithetic {};

struct ControlVariate {
    Real control_mean;   // Known E[control]
    // control_fn computes control variate value from terminal spot
    std::function<Real(Real)> control_fn;
};

struct ImportanceSampling {
    Real drift_shift;    // Shift applied to drift for importance sampling
};

// Monte Carlo pricing engine for European-style payoffs under GBM
class MCEngine {
public:
    struct Config {
        Real spot = 100.0;
        Real rate = 0.05;
        Real sigma = 0.2;
        Real maturity = 1.0;
        Size num_paths = 100000;
        Size num_steps = 1;      // 1 step = terminal only (European)
        uint64_t seed = 42;
    };

    explicit MCEngine(const Config& config);

    // Price with no variance reduction
    MCResult price(const Payoff& payoff) const;

    // Price with antithetic variates
    MCResult price_antithetic(const Payoff& payoff) const;

    // Price with control variate
    // Uses geometric average as control for better accuracy
    MCResult price_control_variate(const Payoff& payoff,
                                    const std::function<Real(Real)>& control_fn,
                                    Real control_mean) const;

    // Price with importance sampling (drift shift for deep OTM)
    MCResult price_importance_sampling(const Payoff& payoff,
                                       Real drift_shift) const;

    const Config& config() const { return config_; }

private:
    Config config_;

    // Generate terminal spot price from a single normal deviate
    Real terminal_spot(Real z) const;

    // Build MCResult from running stats and timer
    MCResult build_result(const RunningStats& stats,
                          std::chrono::steady_clock::time_point start) const;
};

// Builder for fluent configuration
class MCEngineBuilder {
public:
    MCEngineBuilder& spot(Real s) { config_.spot = s; return *this; }
    MCEngineBuilder& rate(Real r) { config_.rate = r; return *this; }
    MCEngineBuilder& sigma(Real v) { config_.sigma = v; return *this; }
    MCEngineBuilder& maturity(Real t) { config_.maturity = t; return *this; }
    MCEngineBuilder& num_paths(Size n) { config_.num_paths = n; return *this; }
    MCEngineBuilder& num_steps(Size n) { config_.num_steps = n; return *this; }
    MCEngineBuilder& seed(uint64_t s) { config_.seed = s; return *this; }

    MCEngine build() const { return MCEngine(config_); }

private:
    MCEngine::Config config_;
};

} // namespace qe
