#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Core
#include "qe/core/types.hpp"
#include "qe/core/constants.hpp"

// Math
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"

// Models
#include "qe/models/black_scholes.hpp"
#include "qe/models/heston.hpp"
#include "qe/models/sabr.hpp"
#include "qe/models/merton_jd.hpp"
#include "qe/models/cir.hpp"

// Monte Carlo
#include "qe/montecarlo/engine.hpp"
#include "qe/instruments/payoff.hpp"
#include "qe/instruments/american.hpp"

// PDE
#include "qe/pde/fdm_solver.hpp"

// Greeks
#include "qe/greeks/finite_difference.hpp"
#include "qe/greeks/pathwise.hpp"
#include "qe/greeks/likelihood_ratio.hpp"

// Curves & Volatility
#include "qe/curves/yield_curve.hpp"
#include "qe/volatility/implied_vol.hpp"
#include "qe/volatility/vol_surface.hpp"

// Risk
#include "qe/risk/var.hpp"
#include "qe/risk/stress.hpp"
#include "qe/risk/portfolio.hpp"

namespace py = pybind11;
using namespace qe;

PYBIND11_MODULE(quant_engine, m) {
    m.doc() = "Quantitative Finance Engine — C++ pricing & risk library with Python bindings";

    // ========================================================================
    // Enums
    // ========================================================================
    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put);

    py::enum_<FDMScheme>(m, "FDMScheme")
        .value("Explicit", FDMScheme::Explicit)
        .value("Implicit", FDMScheme::Implicit)
        .value("CrankNicolson", FDMScheme::CrankNicolson);

    // ========================================================================
    // Black-Scholes
    // ========================================================================
    auto bs = m.def_submodule("bs", "Black-Scholes pricing and Greeks");

    bs.def("call", &bs_call, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"));
    bs.def("put", &bs_put, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"));
    bs.def("price", &bs_price, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"));
    bs.def("delta", &bs_delta, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"));
    bs.def("gamma", &bs_gamma, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"));
    bs.def("vega", &bs_vega, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"));
    bs.def("theta", &bs_theta, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"));
    bs.def("rho", &bs_rho, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"));

    py::class_<BSGreeks>(bs, "Greeks")
        .def_readonly("delta", &BSGreeks::delta)
        .def_readonly("gamma", &BSGreeks::gamma)
        .def_readonly("vega", &BSGreeks::vega)
        .def_readonly("theta", &BSGreeks::theta)
        .def_readonly("rho", &BSGreeks::rho)
        .def("__repr__", [](const BSGreeks& g) {
            return "Greeks(delta=" + std::to_string(g.delta) +
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) +
                   ", theta=" + std::to_string(g.theta) +
                   ", rho=" + std::to_string(g.rho) + ")";
        });

    bs.def("greeks", &bs_greeks, py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"));

    // ========================================================================
    // Monte Carlo Engine
    // ========================================================================
    auto mc = m.def_submodule("mc", "Monte Carlo pricing engine");

    py::class_<MCEngine::Config>(mc, "Config")
        .def(py::init<>())
        .def_readwrite("spot", &MCEngine::Config::spot)
        .def_readwrite("rate", &MCEngine::Config::rate)
        .def_readwrite("sigma", &MCEngine::Config::sigma)
        .def_readwrite("maturity", &MCEngine::Config::maturity)
        .def_readwrite("num_paths", &MCEngine::Config::num_paths)
        .def_readwrite("num_steps", &MCEngine::Config::num_steps)
        .def_readwrite("seed", &MCEngine::Config::seed);

    py::class_<MCResult>(mc, "Result")
        .def_readonly("price", &MCResult::price)
        .def_readonly("std_error", &MCResult::std_error)
        .def_readonly("ci_lo", &MCResult::ci_lo)
        .def_readonly("ci_hi", &MCResult::ci_hi)
        .def_readonly("num_paths", &MCResult::num_paths)
        .def_readonly("elapsed_ms", &MCResult::elapsed_ms)
        .def("__repr__", [](const MCResult& r) {
            return "MCResult(price=" + std::to_string(r.price) +
                   ", SE=" + std::to_string(r.std_error) +
                   ", CI=[" + std::to_string(r.ci_lo) + ", " +
                   std::to_string(r.ci_hi) + "], " +
                   std::to_string(r.elapsed_ms) + "ms)";
        });

    py::class_<Payoff, std::shared_ptr<Payoff>>(mc, "Payoff");

    py::class_<VanillaPayoff, Payoff, std::shared_ptr<VanillaPayoff>>(mc, "VanillaPayoff")
        .def(py::init<Real, OptionType>(), py::arg("strike"), py::arg("type"));

    py::class_<DigitalPayoff, Payoff, std::shared_ptr<DigitalPayoff>>(mc, "DigitalPayoff")
        .def(py::init<Real, OptionType, Real>(),
             py::arg("strike"), py::arg("type"), py::arg("notional") = 1.0);

    py::class_<MCEngine>(mc, "Engine")
        .def(py::init<const MCEngine::Config&>())
        .def("price", &MCEngine::price, py::arg("payoff"))
        .def("price_antithetic", &MCEngine::price_antithetic, py::arg("payoff"));

    // ========================================================================
    // American Options (LSM)
    // ========================================================================
    auto american = m.def_submodule("american", "American option pricing via LSM");

    py::class_<AmericanPricer::Config>(american, "Config")
        .def(py::init<>())
        .def_readwrite("spot", &AmericanPricer::Config::spot)
        .def_readwrite("strike", &AmericanPricer::Config::strike)
        .def_readwrite("rate", &AmericanPricer::Config::rate)
        .def_readwrite("sigma", &AmericanPricer::Config::sigma)
        .def_readwrite("maturity", &AmericanPricer::Config::maturity)
        .def_readwrite("type", &AmericanPricer::Config::type)
        .def_readwrite("num_paths", &AmericanPricer::Config::num_paths)
        .def_readwrite("num_steps", &AmericanPricer::Config::num_steps)
        .def_readwrite("poly_degree", &AmericanPricer::Config::poly_degree)
        .def_readwrite("seed", &AmericanPricer::Config::seed);

    py::class_<AmericanPricer::Result>(american, "Result")
        .def_readonly("price", &AmericanPricer::Result::price)
        .def_readonly("std_error", &AmericanPricer::Result::std_error)
        .def_readonly("early_exercise_premium", &AmericanPricer::Result::early_exercise_premium)
        .def_readonly("european_price", &AmericanPricer::Result::european_price)
        .def_readonly("elapsed_ms", &AmericanPricer::Result::elapsed_ms);

    py::class_<AmericanPricer>(american, "Pricer")
        .def(py::init<const AmericanPricer::Config&>())
        .def("price", &AmericanPricer::price);

    // ========================================================================
    // PDE Solver
    // ========================================================================
    auto pde = m.def_submodule("pde", "Finite difference PDE solvers");

    py::class_<FDMConfig>(pde, "Config")
        .def(py::init<>())
        .def_readwrite("spot", &FDMConfig::spot)
        .def_readwrite("strike", &FDMConfig::strike)
        .def_readwrite("rate", &FDMConfig::rate)
        .def_readwrite("sigma", &FDMConfig::sigma)
        .def_readwrite("maturity", &FDMConfig::maturity)
        .def_readwrite("type", &FDMConfig::type)
        .def_readwrite("n_spot", &FDMConfig::n_spot)
        .def_readwrite("n_time", &FDMConfig::n_time)
        .def_readwrite("spot_min_mult", &FDMConfig::spot_min_mult)
        .def_readwrite("spot_max_mult", &FDMConfig::spot_max_mult);

    py::class_<FDMResult>(pde, "Result")
        .def_readonly("price", &FDMResult::price)
        .def_readonly("delta", &FDMResult::delta)
        .def_readonly("gamma", &FDMResult::gamma)
        .def_readonly("theta", &FDMResult::theta)
        .def_readonly("spot_grid", &FDMResult::spot_grid)
        .def_readonly("option_values", &FDMResult::option_values);

    py::class_<FDMSolver>(pde, "Solver")
        .def(py::init<const FDMConfig&>())
        .def("solve", &FDMSolver::solve, py::arg("scheme"));

    // ========================================================================
    // Heston Model
    // ========================================================================
    auto heston = m.def_submodule("heston", "Heston stochastic volatility model");

    py::class_<HestonParams>(heston, "Params")
        .def(py::init<>())
        .def_readwrite("spot", &HestonParams::spot)
        .def_readwrite("rate", &HestonParams::rate)
        .def_readwrite("v0", &HestonParams::v0)
        .def_readwrite("kappa", &HestonParams::kappa)
        .def_readwrite("theta", &HestonParams::theta)
        .def_readwrite("xi", &HestonParams::xi)
        .def_readwrite("rho", &HestonParams::rho)
        .def_readwrite("maturity", &HestonParams::maturity);

    heston.def("call", &heston_call, py::arg("params"), py::arg("strike"));
    heston.def("put", &heston_put, py::arg("params"), py::arg("strike"));

    py::class_<HestonMC::Config>(heston, "MCConfig")
        .def(py::init<>())
        .def_readwrite("params", &HestonMC::Config::params)
        .def_readwrite("strike", &HestonMC::Config::strike)
        .def_readwrite("type", &HestonMC::Config::type)
        .def_readwrite("num_paths", &HestonMC::Config::num_paths)
        .def_readwrite("num_steps", &HestonMC::Config::num_steps)
        .def_readwrite("seed", &HestonMC::Config::seed);

    py::class_<HestonMC>(heston, "MC")
        .def(py::init<const HestonMC::Config&>())
        .def("price", &HestonMC::price);

    // ========================================================================
    // SABR Model
    // ========================================================================
    auto sabr = m.def_submodule("sabr", "SABR stochastic volatility model");

    py::class_<SABRParams>(sabr, "Params")
        .def(py::init<>())
        .def_readwrite("alpha", &SABRParams::alpha)
        .def_readwrite("beta", &SABRParams::beta)
        .def_readwrite("rho", &SABRParams::rho)
        .def_readwrite("nu", &SABRParams::nu);

    sabr.def("implied_vol", &sabr_implied_vol,
             py::arg("params"), py::arg("forward"), py::arg("strike"), py::arg("T"));
    sabr.def("call", &sabr_call,
             py::arg("params"), py::arg("forward"), py::arg("strike"), py::arg("rate"), py::arg("T"));
    sabr.def("put", &sabr_put,
             py::arg("params"), py::arg("forward"), py::arg("strike"), py::arg("rate"), py::arg("T"));

    // ========================================================================
    // Merton Jump-Diffusion
    // ========================================================================
    auto merton = m.def_submodule("merton", "Merton jump-diffusion model");

    py::class_<MertonParams>(merton, "Params")
        .def(py::init<>())
        .def_readwrite("spot", &MertonParams::spot)
        .def_readwrite("rate", &MertonParams::rate)
        .def_readwrite("sigma", &MertonParams::sigma)
        .def_readwrite("lambda_", &MertonParams::lambda)
        .def_readwrite("mu_j", &MertonParams::mu_j)
        .def_readwrite("sigma_j", &MertonParams::sigma_j)
        .def_readwrite("maturity", &MertonParams::maturity);

    merton.def("call", &merton_call, py::arg("params"), py::arg("strike"),
               py::arg("n_terms") = 50);
    merton.def("put", &merton_put, py::arg("params"), py::arg("strike"),
               py::arg("n_terms") = 50);

    // ========================================================================
    // CIR Model
    // ========================================================================
    auto cir_mod = m.def_submodule("cir", "Cox-Ingersoll-Ross short rate model");

    py::class_<CIRParams>(cir_mod, "Params")
        .def(py::init<>())
        .def_readwrite("r0", &CIRParams::r0)
        .def_readwrite("kappa", &CIRParams::kappa)
        .def_readwrite("theta", &CIRParams::theta)
        .def_readwrite("sigma", &CIRParams::sigma);

    cir_mod.def("bond_price", &cir_bond_price, py::arg("params"), py::arg("T"));
    cir_mod.def("zero_rate", &cir_zero_rate, py::arg("params"), py::arg("T"));
    cir_mod.def("feller_satisfied", &cir_feller_satisfied, py::arg("params"));

    // ========================================================================
    // Yield Curve
    // ========================================================================
    auto curves = m.def_submodule("curves", "Yield curve construction and interpolation");

    py::class_<YieldCurve>(curves, "YieldCurve")
        .def(py::init<const Vec&, const Vec&>(), py::arg("tenors"), py::arg("zero_rates"))
        .def("discount", &YieldCurve::discount, py::arg("T"))
        .def("zero_rate", &YieldCurve::zero_rate, py::arg("T"))
        .def("forward_rate", py::overload_cast<Real>(&YieldCurve::forward_rate, py::const_), py::arg("T"))
        .def("forward_rate", py::overload_cast<Real, Real>(&YieldCurve::forward_rate, py::const_),
             py::arg("T1"), py::arg("T2"))
        .def_property_readonly("tenors", &YieldCurve::tenors)
        .def_property_readonly("rates", &YieldCurve::rates);

    py::class_<DepositRate>(curves, "DepositRate")
        .def(py::init<>())
        .def_readwrite("tenor", &DepositRate::tenor)
        .def_readwrite("rate", &DepositRate::rate);

    py::class_<SwapRate>(curves, "SwapRate")
        .def(py::init<>())
        .def_readwrite("tenor", &SwapRate::tenor)
        .def_readwrite("rate", &SwapRate::rate)
        .def_readwrite("frequency", &SwapRate::frequency);

    curves.def("bootstrap", &bootstrap_curve, py::arg("deposits"), py::arg("swaps"));

    // ========================================================================
    // Implied Volatility & Vol Surface
    // ========================================================================
    auto vol = m.def_submodule("vol", "Implied volatility and vol surface");

    py::class_<ImpliedVolResult>(vol, "ImpliedVolResult")
        .def_readonly("vol", &ImpliedVolResult::vol)
        .def_readonly("iterations", &ImpliedVolResult::iterations)
        .def_readonly("converged", &ImpliedVolResult::converged);

    vol.def("implied", &implied_vol,
            py::arg("market_price"), py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
            py::arg("type"), py::arg("tol") = 1e-8, py::arg("max_iter") = 100);

    py::class_<VolSurface>(vol, "Surface")
        .def(py::init<const Vec&, const Vec&, const Mat&>(),
             py::arg("strikes"), py::arg("maturities"), py::arg("vols"))
        .def("vol", &VolSurface::vol, py::arg("K"), py::arg("T"))
        .def("smile", &VolSurface::smile, py::arg("T"))
        .def_property_readonly("strikes", &VolSurface::strikes)
        .def_property_readonly("maturities", &VolSurface::maturities);

    // ========================================================================
    // Greeks (three methods)
    // ========================================================================
    auto greeks = m.def_submodule("greeks", "Option Greeks via FD, pathwise, and likelihood ratio");

    py::class_<FDGreeks>(greeks, "FDGreeks")
        .def_readonly("delta", &FDGreeks::delta)
        .def_readonly("gamma", &FDGreeks::gamma)
        .def_readonly("vega", &FDGreeks::vega)
        .def_readonly("theta", &FDGreeks::theta)
        .def_readonly("rho", &FDGreeks::rho);

    greeks.def("finite_difference", &mc_fd_greeks,
               py::arg("config"), py::arg("payoff"),
               py::arg("bump_spot") = 1.0, py::arg("bump_sigma") = 0.01,
               py::arg("bump_time") = 1.0/252.0, py::arg("bump_rate") = 0.001);

    py::class_<IPAGreeks>(greeks, "IPAGreeks")
        .def_readonly("delta", &IPAGreeks::delta)
        .def_readonly("vega", &IPAGreeks::vega)
        .def_readonly("rho", &IPAGreeks::rho);

    greeks.def("pathwise", &mc_ipa_greeks,
               py::arg("config"), py::arg("strike"), py::arg("type"));

    py::class_<LRGreeks>(greeks, "LRGreeks")
        .def_readonly("delta", &LRGreeks::delta)
        .def_readonly("vega", &LRGreeks::vega);

    greeks.def("likelihood_ratio", &mc_lr_greeks,
               py::arg("config"), py::arg("payoff"));

    // ========================================================================
    // Risk
    // ========================================================================
    auto risk = m.def_submodule("risk", "VaR, CVaR, stress testing, portfolio risk");

    risk.def("historical_var", [](const Vec& pnl, Real confidence) {
        return historical_var(std::span<const Real>(pnl), confidence);
    }, py::arg("pnl"), py::arg("confidence") = 0.95);

    risk.def("historical_cvar", [](const Vec& pnl, Real confidence) {
        return historical_cvar(std::span<const Real>(pnl), confidence);
    }, py::arg("pnl"), py::arg("confidence") = 0.95);

    risk.def("parametric_var",
             py::overload_cast<Real, Real, Real>(&parametric_var),
             py::arg("mean"), py::arg("std_dev"), py::arg("confidence") = 0.95);

    risk.def("parametric_cvar", &parametric_cvar,
             py::arg("mean"), py::arg("std_dev"), py::arg("confidence") = 0.95);

    py::class_<StressScenario>(risk, "StressScenario")
        .def(py::init<>())
        .def_readwrite("name", &StressScenario::name)
        .def_readwrite("spot_shock", &StressScenario::spot_shock)
        .def_readwrite("vol_shock", &StressScenario::vol_shock)
        .def_readwrite("rate_shock", &StressScenario::rate_shock);

    py::class_<StressResult>(risk, "StressResult")
        .def_readonly("scenario_name", &StressResult::scenario_name)
        .def_readonly("base_value", &StressResult::base_value)
        .def_readonly("stressed_value", &StressResult::stressed_value)
        .def_readonly("pnl", &StressResult::pnl)
        .def_readonly("pnl_pct", &StressResult::pnl_pct);

    risk.def("predefined_scenarios", &predefined_scenarios);

    py::class_<OptionPosition>(risk, "OptionPosition")
        .def(py::init<>())
        .def_readwrite("spot", &OptionPosition::spot)
        .def_readwrite("strike", &OptionPosition::strike)
        .def_readwrite("rate", &OptionPosition::rate)
        .def_readwrite("sigma", &OptionPosition::sigma)
        .def_readwrite("maturity", &OptionPosition::maturity)
        .def_readwrite("type", &OptionPosition::type)
        .def_readwrite("quantity", &OptionPosition::quantity);

    py::class_<PortfolioRisk>(risk, "PortfolioRisk")
        .def_readonly("total_value", &PortfolioRisk::total_value)
        .def_readonly("total_delta", &PortfolioRisk::total_delta)
        .def_readonly("total_gamma", &PortfolioRisk::total_gamma)
        .def_readonly("total_vega", &PortfolioRisk::total_vega)
        .def_readonly("total_theta", &PortfolioRisk::total_theta)
        .def_readonly("total_rho", &PortfolioRisk::total_rho);

    risk.def("portfolio_risk", &portfolio_risk, py::arg("positions"));
    risk.def("stress_test", &stress_test_portfolio,
             py::arg("positions"), py::arg("scenarios"));

    // ========================================================================
    // Statistics utilities
    // ========================================================================
    auto stats = m.def_submodule("stats", "Statistical utilities");

    stats.def("mean", [](const Vec& v) {
        return mean(std::span<const Real>(v));
    });
    stats.def("std_dev", [](const Vec& v) {
        return std_dev(std::span<const Real>(v));
    });
    stats.def("quantile", [](const Vec& v, Real q) {
        return quantile(std::span<const Real>(v), q);
    });

    // Version
    m.attr("__version__") = "0.1.0";
}
