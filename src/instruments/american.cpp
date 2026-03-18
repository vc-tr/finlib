#include "qe/instruments/american.hpp"
#include "qe/models/black_scholes.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace qe {

Real AmericanPricer::intrinsic(Real S) const {
    if (config_.type == OptionType::Call)
        return std::max(S - config_.strike, 0.0);
    return std::max(config_.strike - S, 0.0);
}

Mat AmericanPricer::generate_paths(MersenneTwister& rng) const {
    Size N = config_.num_paths;
    Size M = config_.num_steps;
    Real dt = config_.maturity / static_cast<Real>(M);
    Real drift = (config_.rate - 0.5 * config_.sigma * config_.sigma) * dt;
    Real vol = config_.sigma * std::sqrt(dt);

    Mat paths(N, Vec(M + 1));
    for (Size i = 0; i < N; ++i) {
        paths[i][0] = config_.spot;
        for (Size j = 0; j < M; ++j) {
            Real z = rng.normal();
            paths[i][j + 1] = paths[i][j] * std::exp(drift + vol * z);
        }
    }
    return paths;
}

Vec AmericanPricer::laguerre_basis(Real x, Size degree) const {
    // Weighted Laguerre polynomials: exp(-x/2) * L_n(x)
    // For numerical stability, we use the simple polynomials:
    // L_0 = 1
    // L_1 = 1 - x
    // L_2 = 1 - 2x + x^2/2
    // L_3 = 1 - 3x + 3x^2/2 - x^3/6
    Vec basis(degree + 1);
    basis[0] = 1.0;
    if (degree >= 1) basis[1] = 1.0 - x;
    if (degree >= 2) basis[2] = 1.0 - 2.0 * x + 0.5 * x * x;
    if (degree >= 3) basis[3] = 1.0 - 3.0 * x + 1.5 * x * x - x * x * x / 6.0;
    // Cap at degree 3 for simplicity
    return basis;
}

Vec AmericanPricer::least_squares(const Mat& X, const Vec& Y) const {
    // Normal equations: (X^T X) beta = X^T Y
    // Solve via direct computation (small system, degree+1 unknowns)
    Size n = X.size();     // number of observations
    Size p = X[0].size();  // number of basis functions

    // Compute X^T X
    Mat XtX(p, Vec(p, 0.0));
    for (Size i = 0; i < p; ++i) {
        for (Size j = 0; j < p; ++j) {
            for (Size k = 0; k < n; ++k) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
    }

    // Compute X^T Y
    Vec XtY(p, 0.0);
    for (Size i = 0; i < p; ++i) {
        for (Size k = 0; k < n; ++k) {
            XtY[i] += X[k][i] * Y[k];
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    // Augmented matrix [XtX | XtY]
    Mat aug(p, Vec(p + 1));
    for (Size i = 0; i < p; ++i) {
        for (Size j = 0; j < p; ++j) {
            aug[i][j] = XtX[i][j];
        }
        aug[i][p] = XtY[i];
    }

    // Forward elimination
    for (Size col = 0; col < p; ++col) {
        // Partial pivoting
        Size max_row = col;
        Real max_val = std::abs(aug[col][col]);
        for (Size row = col + 1; row < p; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                max_row = row;
            }
        }
        std::swap(aug[col], aug[max_row]);

        if (std::abs(aug[col][col]) < 1e-15) continue;

        for (Size row = col + 1; row < p; ++row) {
            Real factor = aug[row][col] / aug[col][col];
            for (Size j = col; j <= p; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    Vec beta(p, 0.0);
    for (Size i = p; i > 0; --i) {
        Size row = i - 1;
        Real sum = aug[row][p];
        for (Size j = row + 1; j < p; ++j) {
            sum -= aug[row][j] * beta[j];
        }
        if (std::abs(aug[row][row]) > 1e-15) {
            beta[row] = sum / aug[row][row];
        }
    }

    return beta;
}

AmericanPricer::Result AmericanPricer::price() const {
    auto start = std::chrono::steady_clock::now();

    MersenneTwister rng(config_.seed);
    Size N = config_.num_paths;
    Size M = config_.num_steps;
    Real dt = config_.maturity / static_cast<Real>(M);
    Real discount = std::exp(-config_.rate * dt);

    // Generate all paths
    Mat paths = generate_paths(rng);

    // Cash flow matrix: stores the discounted payoff for each path
    // at the optimal exercise time (initialized to terminal payoff)
    Vec cashflow(N);
    Vec exercise_time(N);
    for (Size i = 0; i < N; ++i) {
        cashflow[i] = intrinsic(paths[i][M]);
        exercise_time[i] = static_cast<Real>(M);
    }

    // Backward induction: at each step, decide exercise vs. continue
    for (Size step = M - 1; step >= 1; --step) {
        // Find in-the-money paths at this step
        std::vector<Size> itm_indices;
        for (Size i = 0; i < N; ++i) {
            if (intrinsic(paths[i][step]) > 0.0) {
                itm_indices.push_back(i);
            }
        }

        if (itm_indices.empty()) continue;

        // Build regression: Y = discounted future cashflow, X = basis(S)
        Size n_itm = itm_indices.size();

        // Discounted continuation values
        Vec Y(n_itm);
        for (Size j = 0; j < n_itm; ++j) {
            Size i = itm_indices[j];
            Real steps_to_exercise = exercise_time[i] - static_cast<Real>(step);
            Y[j] = cashflow[i] * std::pow(discount, steps_to_exercise);
        }

        // Basis matrix
        Mat X(n_itm);
        for (Size j = 0; j < n_itm; ++j) {
            Size i = itm_indices[j];
            Real S_norm = paths[i][step] / config_.strike;  // normalize for stability
            X[j] = laguerre_basis(S_norm, config_.poly_degree);
        }

        // Regress to get continuation value estimate
        Vec beta = least_squares(X, Y);

        // Decision: exercise if intrinsic > estimated continuation
        for (Size j = 0; j < n_itm; ++j) {
            Size i = itm_indices[j];
            Real S_norm = paths[i][step] / config_.strike;
            Vec basis = laguerre_basis(S_norm, config_.poly_degree);

            // Estimated continuation value
            Real continuation = 0.0;
            for (Size k = 0; k < basis.size(); ++k) {
                continuation += beta[k] * basis[k];
            }

            Real exercise_value = intrinsic(paths[i][step]);

            if (exercise_value > continuation) {
                cashflow[i] = exercise_value;
                exercise_time[i] = static_cast<Real>(step);
            }
        }
    }

    // Discount all cashflows to t=0
    RunningStats stats;
    for (Size i = 0; i < N; ++i) {
        Real pv = cashflow[i] * std::pow(discount, exercise_time[i]);
        stats.push(pv);
    }

    // European price for comparison
    Real european = bs_price(config_.spot, config_.strike, config_.rate,
                             config_.sigma, config_.maturity, config_.type);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    return {
        stats.mean(),
        stats.standard_error(),
        stats.mean() - european,
        european,
        elapsed
    };
}

} // namespace qe
