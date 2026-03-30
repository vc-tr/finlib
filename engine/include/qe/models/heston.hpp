#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/montecarlo/engine.hpp"
#include "qe/processes/correlation.hpp"
#include <complex>

namespace qe {

// Heston stochastic volatility model:
// dS = r*S*dt + sqrt(V)*S*dW_1
// dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_2
// Corr(dW_1, dW_2) = rho
struct HestonParams {
    Real spot;
    Real rate;
    Real v0;       // initial variance
    Real kappa;    // mean reversion speed
    Real theta;    // long-run variance
    Real xi;       // vol of vol
    Real rho;      // correlation between spot and vol
    Real maturity;
};

// Semi-analytical pricing via characteristic function (Heston 1993)
// Uses numerical integration (Gauss-Laguerre quadrature)
Real heston_call(const HestonParams& params, Real strike);
Real heston_put(const HestonParams& params, Real strike);

// Monte Carlo pricing under Heston using QE scheme (Andersen 2008)
// The Quadratic-Exponential scheme handles the variance process
// correctly, avoiding negative variance issues
class HestonMC {
public:
    struct Config {
        HestonParams params;
        Real strike;
        OptionType type;
        Size num_paths = 100000;
        Size num_steps = 252;
        uint64_t seed = 42;
    };

    explicit HestonMC(const Config& config) : config_(config) {}

    MCResult price() const;

private:
    Config config_;

    // QE (Quadratic-Exponential) scheme for variance process
    // Prevents negative variance by switching between moment-matched
    // exponential and quadratic approximations
    Real qe_step(Real v, Real dt, Real u_v, Real psi_crit = 1.5) const;
};

// Characteristic function of log-spot under Heston
std::complex<Real> heston_char_func(
    const HestonParams& params, std::complex<Real> u, Real strike);

} // namespace qe
