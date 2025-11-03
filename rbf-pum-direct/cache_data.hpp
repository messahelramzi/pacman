#ifndef CACHE_DATA_HPP
#define CACHE_DATA_HPP

#include <KokkosBlas2_serial_gemv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <misc/ArborX_SymmetricSVD.hpp>

#include "interpolator.hxx"

#define XOR(A, B) (((A) && !(B)) || (!(A) && (B)))

template <typename AViewType, typename diagViewType, typename unitViewType,
          typename bViewType, typename outViewType>
void KOKKOS_FUNCTION solve(AViewType& A, diagViewType& diag, unitViewType& unit,
                           bViewType& b, outViewType& out);

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::prepare_interpolation(void)
{
    ExecSpace execspace{};
    using TensorType = Kokkos::View<Coordinates***, ExecSpace>;
    using MatrixType = Kokkos::View<Coordinates**, ExecSpace>;
    using VectorType = Kokkos::View<Coordinates**, ExecSpace>;

    // NB clusters
    const size_t K = this->_clusters.extent(0);
    // Size of the square matrix NxN
    const size_t N = this->_clusters.extent(1) - 1;
    // Total number of source nodes
    const size_t M = this->_source.extent(0);

    TensorType As("As", K, N, N);
    MatrixType bs("bs", K, N);

    // clang-format off
    /* We copy explicitly the data from *this into the execspace to avoid using a KOKKOS_CLASS_LAMBDA in the kernel below.
     *
     * clusters: A KxN matrix, which contains for each cluster clusters(k,:),
     * the center in clusters(k, 0) and the associated values within the sphere range in clusters(k, 1::)
     *
     * values: A 1xM vector, which contains the values associated to the source points.
     *
     * bounds: A 1xK vector, which contains in bounds(k) the number of nodes to use in the cluster k
     *
     * source: A 1xM vector, which contains the points of the source mesh (used to find the associated value)
     */
    // clang-format on
    auto clusters =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    auto source = Kokkos::create_mirror_view_and_copy(execspace, this->_source);
    auto values = Kokkos::create_mirror_view_and_copy(execspace, this->_values);
    auto bounds = Kokkos::create_mirror_view_and_copy(
        execspace, this->_nb_values_per_cluster);
    auto rbf_function = this->_rbf_function;

    Kokkos::parallel_for(
        "fill rbf system",
        Kokkos::MDRangePolicy(execspace, { 0, 0, 0 }, { K, N, N }),
        KOKKOS_LAMBDA(const size_t& k, const size_t& i, const size_t& j) {
            if (i + 1 < bounds(k) && j + 1 < bounds(k))
            {
                const Point center1 = clusters(k, i + 1);
                const Point center2 = clusters(k, j + 1);
                const Coordinates distance = NDdistance(center1, center2);
                As(k, i, j) = rbf_function(distance);

                size_t value_index = 0;
                for (; value_index < M; ++value_index)
                {
                    if (source(value_index) == center1)
                    {
                        bs(k, i) = values(value_index);
                        break;
                    }
                    if (source(value_index) == center2)
                    {
                        bs(k, j) = values(value_index);
                        break;
                    }
                }
            }
        });
    Kokkos::fence();

    MatrixType coeffs("coeffs", K, N);
    MatrixType diags(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "diags"), K,
        N);
    TensorType units(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "units"), K,
        N, N);

    Kokkos::parallel_for(
        "solve systems", Kokkos::RangePolicy(execspace, 0, K),
        KOKKOS_LAMBDA(const size_t& k) {
            auto A = Kokkos::subview(As, k, Kokkos::ALL(), Kokkos::ALL());
            auto b = Kokkos::subview(bs, k, Kokkos::ALL());
            auto diag = Kokkos::subview(diags, k, Kokkos::ALL());
            auto unit = Kokkos::subview(units, k, Kokkos::ALL(), Kokkos::ALL());
            auto out = Kokkos::subview(coeffs, k, Kokkos::ALL());

            solve(A, diag, unit, b, out);
        });
    Kokkos::fence();

    this->_coeffs =
        MatrixType(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                                      "this->_coeffs"),
                   K, N);
    Kokkos::deep_copy(this->_coeffs, coeffs);
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
