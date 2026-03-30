#include "qe/math/linalg.hpp"
#include <cmath>
#include <stdexcept>

namespace qe {

Mat cholesky(const Mat& A) {
    Size n = A.size();
    if (n == 0) throw std::invalid_argument("cholesky: empty matrix");

    for (const auto& row : A) {
        if (row.size() != n) throw std::invalid_argument("cholesky: matrix must be square");
    }

    Mat L(n, Vec(n, 0.0));

    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j <= i; ++j) {
            Real sum = 0.0;

            if (j == i) {
                // Diagonal element
                for (Size k = 0; k < j; ++k) {
                    sum += L[j][k] * L[j][k];
                }
                Real val = A[j][j] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("cholesky: matrix is not positive definite");
                }
                L[j][j] = std::sqrt(val);
            } else {
                // Off-diagonal element
                for (Size k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

Vec thomas_solve(const Vec& a, const Vec& b, const Vec& c, const Vec& d) {
    Size n = b.size();
    if (n == 0) throw std::invalid_argument("thomas_solve: empty system");
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n) {
        throw std::invalid_argument("thomas_solve: dimension mismatch");
    }

    // Forward sweep
    Vec c_star(n - 1);
    Vec d_star(n);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (Size i = 1; i < n; ++i) {
        Real m = b[i] - a[i - 1] * c_star[i - 1];
        if (i < n - 1) {
            c_star[i] = c[i] / m;
        }
        d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) / m;
    }

    // Back substitution
    Vec x(n);
    x[n - 1] = d_star[n - 1];
    for (Size i = n - 1; i > 0; --i) {
        x[i - 1] = d_star[i - 1] - c_star[i - 1] * x[i];
    }

    return x;
}

Vec mat_vec_mul(const Mat& A, const Vec& x) {
    Size rows = A.size();
    if (rows == 0) throw std::invalid_argument("mat_vec_mul: empty matrix");
    Size cols = A[0].size();
    if (cols != x.size()) throw std::invalid_argument("mat_vec_mul: dimension mismatch");

    Vec y(rows, 0.0);
    for (Size i = 0; i < rows; ++i) {
        for (Size j = 0; j < cols; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

Mat mat_mat_mul(const Mat& A, const Mat& B) {
    Size m = A.size();
    if (m == 0) throw std::invalid_argument("mat_mat_mul: empty matrix A");
    Size n = A[0].size();
    if (n != B.size()) throw std::invalid_argument("mat_mat_mul: dimension mismatch");
    Size p = B[0].size();

    Mat C(m, Vec(p, 0.0));
    for (Size i = 0; i < m; ++i) {
        for (Size k = 0; k < n; ++k) {
            for (Size j = 0; j < p; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat transpose(const Mat& A) {
    if (A.empty()) return {};
    Size m = A.size();
    Size n = A[0].size();

    Mat T(n, Vec(m));
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Mat identity(Size n) {
    Mat I(n, Vec(n, 0.0));
    for (Size i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

} // namespace qe
