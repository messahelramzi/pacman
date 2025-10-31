#ifndef CACHE_DATA_HPP
#define CACHE_DATA_HPP

#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SVD_Serial_Impl.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "interpolator.hxx"

template <typename AViewType, typename uViewType, typename bViewType,
          typename UViewType, typename VtViewType, typename sViewType,
          typename WViewType>
void KOKKOS_FUNCTION solve(AViewType& A, bViewType& b, uViewType& u,
                           UViewType& U, VtViewType& Vt, sViewType& s,
                           WViewType& W);

FULL_TEMPLATE void TEMPLATED_CLASSNAME::prepare_interpolation(void)
{
    assert(this->_clusters.extent(0) > 0 && this->_clusters.extent(1) > 0);
    const ExecSpace execspace{};
    const size_t N = _clusters.extent(1) - 1;
    const size_t M = _clusters.extent(0);
    const size_t Pn = Dim + 1;
    Kokkos::View<Coordinates***, ExecSpace> all_lhs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_lhs"),
        M, N + Pn, N + Pn);
    Kokkos::View<Coordinates**, ExecSpace> all_rhs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_rhs"),
        M, N + Pn);
    auto f = this->_rbf_function;
    auto clusters = Kokkos::create_mirror_view_and_copy(execspace, _clusters);
    auto values = Kokkos::create_mirror_view_and_copy(execspace, _values);
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_lhs rbf "
        "values (A)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU, 0x0LU },
                              { N, N, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            auto rbf_val = f(NDdistance<Dim, Coordinates>(clusters(k, i + 1),
                                                          clusters(k, j + 1)));
            all_lhs(k, i, j) = rbf_val;
            all_lhs(k, j, i) = rbf_val;
            all_rhs(k, i) = values(i);
        });
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs poly "
        "values (P/Pt)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU }, { N, M }),
        KOKKOS_LAMBDA(const size_t& j, const size_t& k) {
            // Constant contribution
            all_lhs(k, N, j) = 1.0;
            all_lhs(k, j, N) = 1.0;
            for (size_t i = 0; i < Dim; ++i)
            {
                all_lhs(k, N + i + 1, j) = clusters(k, j)[i];
                all_lhs(k, j, N + i + 1) = clusters(k, j)[i];
            }
        });
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs padding "
        "zeros (0)",
        Kokkos::MDRangePolicy(execspace, { N, N, 0x0LU },
                              { N + Pn, N + Pn, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            all_lhs(k, i, j) = 0.0;
            all_rhs(k, i) = 0.0;
        });
    Kokkos::fence();

    Kokkos::View<Coordinates**, ExecSpace> coeffs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "coeffs"), M,
        N + Pn);

    Kokkos::View<Coordinates***, ExecSpace> all_U(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "U"), M,
        N + Pn, N + Pn);
    Kokkos::View<Coordinates***, ExecSpace> all_Vt(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "Vt"), M,
        N + Pn, N + Pn);
    Kokkos::View<Coordinates**, ExecSpace> all_s(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "s"), M,
        N + Pn);
    Kokkos::View<Coordinates**, Kokkos::LayoutRight, ExecSpace> all_W(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "W"), M,
        N + Pn);

    Kokkos::parallel_for(
        "solve all systems", Kokkos::RangePolicy(execspace, 0, M),
        KOKKOS_LAMBDA(const size_t& i) {
            auto A = Kokkos::subview(all_lhs, i, Kokkos::ALL(), Kokkos::ALL());
            auto b = Kokkos::subview(all_rhs, i, Kokkos::ALL());
            auto u = Kokkos::subview(coeffs, i, Kokkos::ALL());
            auto U = Kokkos::subview(all_U, i, Kokkos::ALL(), Kokkos::ALL());
            auto Vt = Kokkos::subview(all_Vt, i, Kokkos::ALL(), Kokkos::ALL());
            auto s = Kokkos::subview(all_s, i, Kokkos::ALL());
            auto W = Kokkos::subview(all_W, i, Kokkos::ALL());

            solve(A, b, u, U, Vt, s, W);
        });
    Kokkos::fence();
    this->_coeffs = Kokkos::View<Coordinates**, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_coeffs"),
        M, N + Pn);
    Kokkos::deep_copy(this->_coeffs, coeffs);
}

/* Solves Au=b for u
 * @param A: [in] NxN matrix, symmetric, can be singular
 * @param b: [in] Nx1 vector, the values can be anything
 * @param u: [out] Nx1 vector, output param, solution of the system
 * @param U: [out] NxN matrix, left singular vectors in cols
 * @param Vt: [out] NxN matrix, right singular vectors in rows
 * @param s: [out] Nx1 vector, singular values
 * @param W: [in] Nx1 vector, temp workspace
 */
template <typename AViewType, typename uViewType, typename bViewType,
          typename UViewType, typename VtViewType, typename sViewType,
          typename WViewType>
void KOKKOS_FUNCTION solve(AViewType& A, bViewType& b, uViewType& u,
                           UViewType& U, VtViewType& Vt, sViewType& s,
                           WViewType& W)
{
    using ValueType = typename AViewType::value_type;
    const int N = (int)A.extent(0); // square matrix

    // 1) Compute full SVD: A = U * diag(s) * Vt
    KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag{},
                                     A, // overwritten
                                     U, // output U
                                     s, // singular values
                                     Vt, // output Vt
                                     W // workspace
    );

    // 2) Compute tmp = U^T * b
    for (int i = 0; i < N; ++i)
    {
        ValueType sum = 0;
        for (int j = 0; j < N; ++j)
            sum += U(j, i) * b(j);
        W(i) = sum;
    }

    // 3) Scale by 1/s (pseudo-inverse), with tolerance
    const ValueType tol = 1e-12;
    for (int i = 0; i < N; ++i)
    {
        if (s[i] > tol)
            W(i) /= s(i);
        else
            W(i) = static_cast<ValueType>(0);
    }

    // 4) Compute u = V * tmp
    for (int i = 0; i < N; ++i)
    {
        ValueType sum = 0;
        for (int j = 0; j < N; ++j)
        {
            sum += Vt(j, i) * W(j); // V = Vt^T
        }
        u(i) = sum;
    }
}

#endif /* ! CACHE_DATA_HPP */
