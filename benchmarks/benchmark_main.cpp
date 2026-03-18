#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "qe/models/black_scholes.hpp"
#include "qe/montecarlo/engine.hpp"
#include "qe/instruments/payoff.hpp"
#include "qe/instruments/american.hpp"
#include "qe/pde/fdm_solver.hpp"
#include "qe/models/heston.hpp"
#include "qe/models/sabr.hpp"
#include "qe/models/merton_jd.hpp"
#include "qe/volatility/implied_vol.hpp"
#include "qe/risk/var.hpp"
#include "qe/risk/portfolio.hpp"
#include "qe/risk/stress.hpp"
#include "qe/math/random.hpp"

using namespace qe;

struct BenchResult {
    std::string name;
    double avg_us;       // microseconds
    Size iterations;
    std::string extra;   // e.g. price result
};

template<typename Func>
BenchResult benchmark(const std::string& name, Func func, Size iters = 1000) {
    // Warmup
    for (Size i = 0; i < 3; ++i) func();

    auto start = std::chrono::steady_clock::now();
    double result_val = 0.0;
    for (Size i = 0; i < iters; ++i) {
        result_val = func();
    }
    auto end = std::chrono::steady_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();

    return {name, total_us / static_cast<double>(iters), iters,
            std::to_string(result_val)};
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    QUANT ENGINE BENCHMARKS                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<BenchResult> results;

    // ========================================================================
    // Black-Scholes analytical
    // ========================================================================
    std::cout << "  Black-Scholes Analytical\n";
    std::cout << "  ────────────────────────\n";

    results.push_back(benchmark("BS Call Price", []() {
        return bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    }, 1000000));

    results.push_back(benchmark("BS Put Price", []() {
        return bs_put(100.0, 100.0, 0.05, 0.2, 1.0);
    }, 1000000));

    results.push_back(benchmark("BS All Greeks", []() {
        auto g = bs_greeks(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call);
        return g.delta;
    }, 1000000));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // Monte Carlo
    // ========================================================================
    std::cout << "\n  Monte Carlo Pricing\n";
    std::cout << "  ───────────────────\n";
    results.clear();

    auto mc_bench = [](Size n_paths) {
        MCEngine::Config cfg;
        cfg.spot = 100; cfg.rate = 0.05; cfg.sigma = 0.2; cfg.maturity = 1.0;
        cfg.num_paths = n_paths; cfg.seed = 42;
        VanillaPayoff payoff(100.0, OptionType::Call);
        MCEngine engine(cfg);
        return engine.price(payoff).price;
    };

    results.push_back(benchmark("MC 10K paths", [&]() { return mc_bench(10000); }, 100));
    results.push_back(benchmark("MC 100K paths", [&]() { return mc_bench(100000); }, 10));
    results.push_back(benchmark("MC 1M paths", [&]() { return mc_bench(1000000); }, 3));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // American LSM
    // ========================================================================
    std::cout << "\n  American Options (LSM)\n";
    std::cout << "  ──────────────────────\n";
    results.clear();

    results.push_back(benchmark("LSM 50K paths", []() {
        AmericanPricer::Config cfg;
        cfg.spot = 100; cfg.strike = 100; cfg.rate = 0.05; cfg.sigma = 0.2;
        cfg.maturity = 1.0; cfg.type = OptionType::Put;
        cfg.num_paths = 50000; cfg.num_steps = 50; cfg.seed = 42;
        return AmericanPricer(cfg).price().price;
    }, 3));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // PDE
    // ========================================================================
    std::cout << "\n  PDE (Finite Difference)\n";
    std::cout << "  ───────────────────────\n";
    results.clear();

    results.push_back(benchmark("Crank-Nicolson 200x1000", []() {
        FDMConfig cfg;
        cfg.spot = 100; cfg.strike = 100; cfg.rate = 0.05; cfg.sigma = 0.2;
        cfg.maturity = 1.0; cfg.type = OptionType::Call;
        cfg.n_spot = 200; cfg.n_time = 1000;
        return FDMSolver(cfg).solve(FDMScheme::CrankNicolson).price;
    }, 50));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // Heston
    // ========================================================================
    std::cout << "\n  Heston Stochastic Vol\n";
    std::cout << "  ─────────────────────\n";
    results.clear();

    HestonParams hp;
    hp.spot = 100; hp.rate = 0.05; hp.v0 = 0.04; hp.kappa = 2.0;
    hp.theta = 0.04; hp.xi = 0.3; hp.rho = -0.7; hp.maturity = 1.0;

    results.push_back(benchmark("Heston Analytical", [&]() {
        return heston_call(hp, 100.0);
    }, 10000));

    results.push_back(benchmark("Heston MC 100K", [&]() {
        HestonMC::Config mc_cfg;
        mc_cfg.params = hp; mc_cfg.strike = 100; mc_cfg.type = OptionType::Call;
        mc_cfg.num_paths = 100000; mc_cfg.num_steps = 252; mc_cfg.seed = 42;
        return HestonMC(mc_cfg).price().price;
    }, 3));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // Implied Vol
    // ========================================================================
    std::cout << "\n  Implied Volatility Solver\n";
    std::cout << "  ────────────────────────\n";
    results.clear();

    Real target_price = bs_call(100.0, 100.0, 0.05, 0.25, 1.0);
    results.push_back(benchmark("Newton-Raphson IV", [&]() {
        return implied_vol(target_price, 100.0, 100.0, 0.05, 1.0, OptionType::Call).vol;
    }, 100000));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    // ========================================================================
    // Risk
    // ========================================================================
    std::cout << "\n  Risk Metrics\n";
    std::cout << "  ────────────\n";
    results.clear();

    MersenneTwister rng(42);
    Vec pnl(100000);
    for (auto& x : pnl) x = rng.normal() * 0.02;

    results.push_back(benchmark("VaR 100K samples", [&]() {
        return historical_var(std::span<const Real>(pnl), 0.95);
    }, 100));

    results.push_back(benchmark("CVaR 100K samples", [&]() {
        return historical_cvar(std::span<const Real>(pnl), 0.95);
    }, 100));

    std::vector<OptionPosition> portfolio;
    for (int i = 0; i < 20; ++i) {
        portfolio.push_back({100.0, 90.0 + static_cast<Real>(i) * 2.0,
                            0.05, 0.2, 1.0,
                            (i % 2 == 0) ? OptionType::Call : OptionType::Put,
                            static_cast<Real>(10 - i)});
    }

    results.push_back(benchmark("Portfolio risk (20 pos)", [&]() {
        return portfolio_risk(portfolio).total_value;
    }, 10000));

    results.push_back(benchmark("Stress test (20 pos, 7 scen)", [&]() {
        auto scenarios = predefined_scenarios();
        auto res = stress_test_portfolio(portfolio, scenarios);
        return res[0].pnl;
    }, 1000));

    for (const auto& r : results) {
        std::cout << "    " << std::left << std::setw(25) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0)
                  << r.avg_us << " μs/call"
                  << "   (" << r.iterations << " iters)\n";
    }

    std::cout << "\n  Done.\n\n";
    return 0;
}
