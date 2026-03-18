#include "qe/models/heston.hpp"
#include "qe/models/black_scholes.hpp"
#include "qe/core/constants.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace qe {

// ============================================================================
// Heston characteristic function (Schoutens formulation, numerically stable)
// phi(u) = E[exp(i*u*ln(S(T)/K))]
// ============================================================================
std::complex<Real> heston_char_func(
    const HestonParams& p, std::complex<Real> u, Real strike)
{
    using C = std::complex<Real>;
    C i(0.0, 1.0);

    Real tau = p.maturity;
    Real x = std::log(p.spot / strike);

    C d = std::sqrt(
        (p.rho * p.xi * i * u - p.kappa) * (p.rho * p.xi * i * u - p.kappa)
        + p.xi * p.xi * (i * u + u * u)
    );

    C g = (p.kappa - p.rho * p.xi * i * u - d)
        / (p.kappa - p.rho * p.xi * i * u + d);

    C exp_d_tau = std::exp(-d * tau);

    C A = i * u * x
        + (p.kappa * p.theta / (p.xi * p.xi))
          * ((p.kappa - p.rho * p.xi * i * u - d) * tau
             - 2.0 * std::log((1.0 - g * exp_d_tau) / (1.0 - g)));

    C B = ((p.kappa - p.rho * p.xi * i * u - d) / (p.xi * p.xi))
        * (1.0 - exp_d_tau) / (1.0 - g * exp_d_tau);

    return std::exp(A + B * p.v0);
}

// ============================================================================
// Semi-analytical pricing via numerical integration
// C = exp(-r*T) * [0.5*(F-K) + (1/pi) * integral]
// where integral uses the characteristic function
// Uses Gauss-Laguerre quadrature for semi-infinite integral
// ============================================================================

// Gauss-Laguerre nodes and weights (32 points)
static const Size GL_N = 32;
static const Real gl_nodes[] = {
    0.04448936583326, 0.23452610952245, 0.57688462930188, 1.07244875381782,
    1.72240877644465, 2.52833670642579, 3.49221327302199, 4.61645676974976,
    5.90395849498530, 7.35812673318624, 8.98294092421259, 10.78301863254752,
    12.76369798168742, 14.93113975552419, 17.29245433671532, 19.85586156227399,
    22.63089499387429, 25.62862590791844, 28.86210080913696, 32.34662915396476,
    36.10032294817465, 40.14531807709713, 44.50920799575495, 49.22478089653567,
    54.33372133339038, 59.89206290266734, 65.97539280752850, 72.68762809066271,
    80.18744697791352, 88.73534824499788, 98.82954286828507, 111.75139809793770
};

static const Real gl_weights[] = {
    0.10921834195239, 0.21044310793881, 0.23521322966984, 0.19590333597289,
    0.12998378628607, 0.07057862386571, 0.03176091250917, 0.01191821483484,
    0.00373881629461, 0.00098080331002, 0.00021486491880, 0.00003920341967,
    0.00000593278702, 0.00000074028375, 0.00000007577280, 0.00000000630281,
    0.00000000042131, 0.00000000002237, 0.00000000000092, 0.00000000000003,
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
    0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000
};

Real heston_call(const HestonParams& p, Real K) {
    using C = std::complex<Real>;
    C i(0.0, 1.0);

    Real x = std::log(p.spot / K);
    Real discount = std::exp(-p.rate * p.maturity);

    // Lewis (2001) formula for call price:
    // C = S - K*exp(-r*T) * (1/pi) * integral_0^inf Re[exp(-iu*x) * phi(u-i/2) / (u^2+1/4)] du
    // where phi is the characteristic function of log(S(T)/S(0)) under risk-neutral

    // Alternatively, use the Gil-Pelaez inversion:
    // P1 = 0.5 + (1/pi) * integral Re[exp(-iu*ln(K)) * phi_1(u) / (iu)] du
    // P2 = 0.5 + (1/pi) * integral Re[exp(-iu*ln(K)) * phi_2(u) / (iu)] du
    // C = S*P1 - K*exp(-rT)*P2

    auto integrand_P1 = [&](Real u) -> Real {
        if (u < 1e-10) return 0.0;
        C phi = heston_char_func(p, C(u, -1.0), 1.0);  // phi(u - i)
        C phi_0 = heston_char_func(p, C(0.0, -1.0), 1.0);  // normalization
        C result = std::exp(-i * u * std::log(K)) * phi / (i * u * phi_0);
        return result.real();
    };

    auto integrand_P2 = [&](Real u) -> Real {
        if (u < 1e-10) return 0.0;
        C phi = heston_char_func(p, C(u, 0.0), 1.0);
        C result = std::exp(-i * u * std::log(K)) * phi / (i * u);
        return result.real();
    };

    // Integrate using Gauss-Laguerre (substitution for semi-infinite integral)
    Real integral_P1 = 0.0, integral_P2 = 0.0;
    for (Size j = 0; j < GL_N; ++j) {
        Real u = gl_nodes[j];
        Real w = gl_weights[j] * std::exp(gl_nodes[j]);  // undo Laguerre weight
        integral_P1 += w * integrand_P1(u);
        integral_P2 += w * integrand_P2(u);
    }

    Real P1 = 0.5 + integral_P1 / PI;
    Real P2 = 0.5 + integral_P2 / PI;

    return std::max(p.spot * P1 - K * discount * P2, 0.0);
}

Real heston_put(const HestonParams& p, Real K) {
    Real call = heston_call(p, K);
    Real discount = std::exp(-p.rate * p.maturity);
    return call - p.spot + K * discount;  // put-call parity
}

// ============================================================================
// Heston MC with full truncation Euler scheme
// V+ = max(V, 0) used in diffusion to prevent negative variance
// ============================================================================

Real HestonMC::qe_step(Real v, Real dt, Real u_v, Real /*psi_crit*/) const {
    const auto& p = config_.params;
    Real v_pos = std::max(v, 0.0);

    // Euler step for variance with full truncation
    Real dv = p.kappa * (p.theta - v_pos) * dt + p.xi * std::sqrt(v_pos * dt) * u_v;
    return std::max(v + dv, 0.0);
}

MCResult HestonMC::price() const {
    auto start = std::chrono::steady_clock::now();
    const auto& p = config_.params;

    MersenneTwister rng(config_.seed);
    Real dt = p.maturity / static_cast<Real>(config_.num_steps);
    Real sqrt_dt = std::sqrt(dt);
    Real discount = std::exp(-p.rate * p.maturity);

    RunningStats stats;

    for (Size i = 0; i < config_.num_paths; ++i) {
        Real S = p.spot;
        Real V = p.v0;

        for (Size j = 0; j < config_.num_steps; ++j) {
            Real z1 = rng.normal();
            Real z2 = rng.normal();
            auto [w_s, w_v] = correlate(z1, z2, p.rho);

            Real V_pos = std::max(V, 0.0);

            // Log-Euler for spot
            S = S * std::exp((p.rate - 0.5 * V_pos) * dt
                              + std::sqrt(V_pos) * sqrt_dt * w_s);

            // Euler for variance with full truncation
            V = V + p.kappa * (p.theta - V_pos) * dt
                  + p.xi * std::sqrt(V_pos) * sqrt_dt * w_v;
            V = std::max(V, 0.0);
        }

        Real payoff;
        if (config_.type == OptionType::Call) {
            payoff = std::max(S - config_.strike, 0.0);
        } else {
            payoff = std::max(config_.strike - S, 0.0);
        }

        stats.push(discount * payoff);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Real se = stats.standard_error();

    return {stats.mean(), se, stats.mean() - 1.96 * se,
            stats.mean() + 1.96 * se, stats.count(), elapsed};
}

} // namespace qe
