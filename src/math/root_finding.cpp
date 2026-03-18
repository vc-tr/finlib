#include "qe/math/root_finding.hpp"
#include <cmath>
#include <stdexcept>

namespace qe {

RootResult newton_raphson(
    const std::function<Real(Real)>& f,
    const std::function<Real(Real)>& df,
    Real x0,
    Real tol,
    Size max_iter
) {
    Real x = x0;

    for (Size i = 0; i < max_iter; ++i) {
        Real fx = f(x);
        Real dfx = df(x);

        if (std::abs(dfx) < 1e-30) {
            return {x, i, false};  // derivative too small
        }

        Real dx = fx / dfx;
        x -= dx;

        if (std::abs(dx) < tol) {
            return {x, i + 1, true};
        }
    }

    return {x, max_iter, false};
}

RootResult brent(
    const std::function<Real(Real)>& f,
    Real a,
    Real b,
    Real tol,
    Size max_iter
) {
    Real fa = f(a);
    Real fb = f(b);

    if (fa * fb > 0.0) {
        throw std::invalid_argument("brent: f(a) and f(b) must have opposite signs");
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    Real c = a;
    Real fc = fa;
    bool mflag = true;
    Real s = 0.0;
    Real d = 0.0;

    for (Size i = 0; i < max_iter; ++i) {
        if (std::abs(fb) < tol || std::abs(b - a) < tol) {
            return {b, i + 1, true};
        }

        if (std::abs(fa - fc) > 1e-30 && std::abs(fb - fc) > 1e-30) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for accepting s
        Real mid = (a + b) / 2.0;
        bool cond1 = !((s > std::min((3.0 * a + b) / 4.0, b)) &&
                        (s < std::max((3.0 * a + b) / 4.0, b)));
        bool cond2 = mflag && (std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = !mflag && (std::abs(s - b) >= std::abs(c - d) / 2.0);
        bool cond4 = mflag && (std::abs(b - c) < tol);
        bool cond5 = !mflag && (std::abs(c - d) < tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = mid;
            mflag = true;
        } else {
            mflag = false;
        }

        Real fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    return {b, max_iter, false};
}

RootResult bisection(
    const std::function<Real(Real)>& f,
    Real a,
    Real b,
    Real tol,
    Size max_iter
) {
    Real fa = f(a);
    Real fb = f(b);

    if (fa * fb > 0.0) {
        throw std::invalid_argument("bisection: f(a) and f(b) must have opposite signs");
    }

    for (Size i = 0; i < max_iter; ++i) {
        Real mid = (a + b) / 2.0;
        Real fm = f(mid);

        if (std::abs(fm) < tol || (b - a) / 2.0 < tol) {
            return {mid, i + 1, true};
        }

        if (fa * fm < 0.0) {
            b = mid;
            fb = fm;
        } else {
            a = mid;
            fa = fm;
        }
    }

    return {(a + b) / 2.0, max_iter, false};
}

} // namespace qe
