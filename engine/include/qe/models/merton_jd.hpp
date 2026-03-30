#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/montecarlo/engine.hpp"
#include "qe/models/black_scholes.hpp"

namespace qe {

// Merton (1976) jump-diffusion model:
// dS/S = (r - lambda*k)*dt + sigma*dW + J*dN
// where N is Poisson with intensity lambda,
// and J = exp(mu_j + sigma_j * Z) - 1 (lognormal jumps)
// k = E[J] = exp(mu_j + sigma_j^2/2) - 1
struct MertonParams {
    Real spot;
    Real rate;
    Real sigma;     // diffusion volatility
    Real lambda;    // jump intensity (arrivals per year)
    Real mu_j;      // mean log-jump size
    Real sigma_j;   // std of log-jump size
    Real maturity;
};

// Semi-analytical: sum of Black-Scholes prices weighted by Poisson probs
// C_Merton = sum_{n=0}^{N} (e^{-lambda'*T} * (lambda'*T)^n / n!) * BS(S, K, r_n, sigma_n, T)
Real merton_call(const MertonParams& params, Real strike, Size n_terms = 50);
Real merton_put(const MertonParams& params, Real strike, Size n_terms = 50);

// Monte Carlo pricing under Merton jump-diffusion
MCResult merton_mc(const MertonParams& params, Real strike, OptionType type,
                   Size num_paths = 100000, Size num_steps = 252, uint64_t seed = 42);

} // namespace qe
