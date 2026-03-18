#pragma once

#include "qe/core/types.hpp"
#include "qe/math/statistics.hpp"
#include <vector>

namespace qe {

// Track convergence of MC estimator at specified sample sizes
struct ConvergencePoint {
    Size n_paths;
    Real estimate;
    Real std_error;
    Real ci_lo;
    Real ci_hi;
};

using ConvergenceTable = std::vector<ConvergencePoint>;

// Build convergence table by recording stats at powers of 2
class ConvergenceMonitor {
public:
    explicit ConvergenceMonitor(Real true_value = 0.0)
        : true_value_(true_value), next_checkpoint_(64) {}

    void push(Real sample) {
        stats_.push(sample);
        if (stats_.count() == next_checkpoint_) {
            record();
            next_checkpoint_ *= 2;
        }
    }

    // Force recording current state
    void finalize() {
        if (table_.empty() || table_.back().n_paths != stats_.count()) {
            record();
        }
    }

    const ConvergenceTable& table() const { return table_; }
    Real true_value() const { return true_value_; }

private:
    RunningStats stats_;
    Real true_value_;
    Size next_checkpoint_;
    ConvergenceTable table_;

    void record() {
        Real se = stats_.standard_error();
        table_.push_back({
            stats_.count(),
            stats_.mean(),
            se,
            stats_.mean() - 1.96 * se,
            stats_.mean() + 1.96 * se
        });
    }
};

} // namespace qe
