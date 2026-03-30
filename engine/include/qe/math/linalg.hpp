#pragma once

#include "qe/core/types.hpp"

namespace qe {

// Cholesky decomposition: A = L * L^T
// Input: symmetric positive-definite matrix A (n x n)
// Returns: lower triangular matrix L
Mat cholesky(const Mat& A);

// Solve tridiagonal system using Thomas algorithm
// a: sub-diagonal (n-1 elements)
// b: main diagonal (n elements)
// c: super-diagonal (n-1 elements)
// d: right-hand side (n elements)
// Returns: solution vector x (n elements)
Vec thomas_solve(const Vec& a, const Vec& b, const Vec& c, const Vec& d);

// Matrix-vector multiplication: y = A * x
Vec mat_vec_mul(const Mat& A, const Vec& x);

// Matrix-matrix multiplication: C = A * B
Mat mat_mat_mul(const Mat& A, const Mat& B);

// Matrix transpose
Mat transpose(const Mat& A);

// Identity matrix
Mat identity(Size n);

} // namespace qe
