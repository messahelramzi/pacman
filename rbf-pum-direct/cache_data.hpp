#ifndef CACHE_DATA_HPP
#define CACHE_DATA_HPP

#include <KokkosBlas.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <misc/ArborX_SymmetricSVD.hpp>

#include "interpolator.hxx"

template <typename AViewType, typename diagViewType, typename unitViewType,
          typename bViewType, typename outViewType>
void KOKKOS_FUNCTION solve(AViewType& A, diagViewType& diag, unitViewType& unit,
                           bViewType& b, outViewType& out);

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
    auto clusters =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    auto values = Kokkos::create_mirror_view_and_copy(execspace, this->_values);
    auto nb_nodes = Kokkos::create_mirror_view_and_copy(
        execspace, this->_nb_values_per_cluster);
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_lhs rbf "
        "values (A)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU, 0x0LU },
                              { N, N, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            if (i < nb_nodes(k) && j < nb_nodes(k))
            {
                auto rbf_val = f(NDdistance<Dim, Coordinates>(
                    clusters(k, i + 1), clusters(k, j + 1)));
                all_lhs(k, i, j) = rbf_val;
                all_rhs(k, i) = values(i);
            }
            else
            {
                all_lhs(k, i, j) = 0.0;
                all_rhs(k, i) = 0.0;
            }
        });
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_lhs poly "
        "values (P/Pt)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU }, { N, M }),
        KOKKOS_LAMBDA(const size_t& j, const size_t& k) {
            if (j < nb_nodes(k))
            {
                // Constant contribution
                all_lhs(k, j, N) = 1.0;
                all_lhs(k, N, j) = 1.0;
                for (int axis = 0; axis < Dim; ++axis)
                {
                    all_lhs(k, j, N + 1 + axis) = clusters(k, j)[axis];
                    all_lhs(k, N + 1 + axis, j) = clusters(k, j)[axis];
                }
            }
            else
            {
                all_lhs(k, j, N) = 0.0;
                all_lhs(k, N, j) = 0.0;
                for (int axis = 0; axis < Dim; ++axis)
                {
                    all_lhs(k, j, N + 1 + axis) = 0.0;
                    all_lhs(k, N + 1 + axis, j) = 0.0;
                }
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

    auto m1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all_lhs);
    auto m2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all_rhs);
    for (size_t i = 0; i < M; ++i)
    {
        auto mat = Kokkos::subview(m1, i, Kokkos::ALL(), Kokkos::ALL());
        auto val = Kokkos::subview(m2, i, Kokkos::ALL());
        print_2d_view(mat);
        print_view(val);
    }

    Kokkos::View<Coordinates**, ExecSpace> coeffs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "coeffs"), M,
        N + Pn);

    Kokkos::View<Coordinates**, ExecSpace> all_diag(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "all_diag"),
        M, N + Pn);
    Kokkos::View<Coordinates***, ExecSpace> all_unit(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "all_unit"),
        M, N + Pn, N + Pn);

    Kokkos::parallel_for(
        "solve all systems", Kokkos::RangePolicy(execspace, 0, M),
        KOKKOS_LAMBDA(const size_t& i) {
            // clang-format off
            auto    A = Kokkos::subview(all_lhs, i, Kokkos::ALL(), Kokkos::ALL());
            auto    b = Kokkos::subview(all_rhs, i, Kokkos::ALL());
            auto diag = Kokkos::subview(all_diag, i, Kokkos::ALL());
            auto unit = Kokkos::subview(all_unit, i, Kokkos::ALL(), Kokkos::ALL());
            auto  out = Kokkos::subview(coeffs, i, Kokkos::ALL());
            solve(A, diag, unit, b, out);
            // clang-format on
        });
    Kokkos::fence();
    this->_coeffs = Kokkos::View<Coordinates**, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_coeffs"),
        M, N + Pn);
    Kokkos::deep_copy(this->_coeffs, coeffs);
    print_2d_view(this->_coeffs, "\n");
}

/* Solves Au=b for u
 * @warning ALL OF THE PARAMS CAN BE MODIFIED DURING THE CALL
 * @param A: [in] NxN matrix, symmetric, can be singular
 * @param diag: [in] Nx1 vector, the values can be anything
 * @param unit: [in] Nx1 vector, output param, solution of the system
 * @param b: [in] NxN matrix, left singular vectors in cols
 * @param out: [out] NxN matrix, right singular vectors in rows
 */
template <typename AViewType, typename diagViewType, typename unitViewType,
          typename bViewType, typename outViewType>
void KOKKOS_FUNCTION solve(AViewType& A, diagViewType& diag, unitViewType& unit,
                           bViewType& b, outViewType& out)
{
    // 1. We compute the pseudo inverse of A.
    ArborX::Details::symmetricPseudoInverseSVDKernel(A, diag, unit);
    // 2. We compute u = A^{-1}b
    KokkosBlas::Experimental::serial_gemv(0x4E, 1.0, A, b, 0.0, out);
}

#endif /* ! CACHE_DATA_HPP */
