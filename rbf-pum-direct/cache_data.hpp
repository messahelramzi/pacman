#ifndef CACHE_DATA_HPP
#define CACHE_DATA_HPP

#include <KokkosBlas2_serial_gemv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <misc/ArborX_SymmetricSVD.hpp>

#include "interpolator.hxx"

template <typename member_type, typename AViewType, typename bViewType,
          typename outViewType>
void KOKKOS_INLINE_FUNCTION matrix_vector_product(const member_type& member,
                                                  AViewType& A, bViewType& b,
                                                  outViewType& out);

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::prepare_interpolation(void)
{
    const std::string _region_name =
        "RbfPumInterpolator::prepare_interpolation";
    Kokkos::Profiling::pushRegion(_region_name);
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

    Kokkos::Profiling::pushRegion(_region_name + "::allocate systems data");
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
    Kokkos::Profiling::popRegion(); // ! Allocate Systems Data

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using ScratchView =
        Kokkos::View<Point*, typename ExecSpace::scratch_memory_space>;
    size_t scratch_size = ScratchView::shmem_size(N);
    int level = 0;

    Kokkos::parallel_for(
        "fill rbf system team policy",
        team_policy(ExecSpace{}, K, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(member_type const& member) {
            const size_t k = member.league_rank();
            ScratchView local_centers(member.team_scratch(level), N);
            Kokkos::parallel_for(Kokkos::TeamVectorRange(member, N),
                                 [=](const size_t& p) {
                                     local_centers(p) = clusters(k, p + 1);
                                 });
            member.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadMDRange(member, N, N),
                                 [=](const size_t& i, const size_t& j) {
                                     As(k, i, j) = rbf_function(NDdistance(
                                         local_centers(i), local_centers(j)));
                                 });
        });
    Kokkos::parallel_for(
        "fill rbf rhs team policy",
        team_policy(ExecSpace{}, K, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(member_type const& member) {
            const size_t k = member.league_rank();
            ScratchView local_centers(member.team_scratch(level), N);
            Kokkos::parallel_for(Kokkos::TeamVectorRange(member, N),
                                 [=](const size_t& p) {
                                     local_centers(p) = clusters(k, p + 1);
                                 });
            member.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadMDRange(member, N, M),
                                 [=](const size_t& i, const size_t& j) {
                                     if (local_centers(i) == source(j))
                                     {
                                         bs(k, i) = values(j);
                                     }
                                 });
        });
    Kokkos::fence();

    MatrixType coeffs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "coeffs"), K,
        N);
    MatrixType diags(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "diags"), K,
        N);
    TensorType units(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "units"), K,
        N, N);

    Kokkos::parallel_for(
        "compute inverse with svd", Kokkos::RangePolicy(execspace, 0, K),
        KOKKOS_LAMBDA(const size_t& k) {
            auto A = Kokkos::subview(As, k, Kokkos::ALL(), Kokkos::ALL());
            auto diag = Kokkos::subview(diags, k, Kokkos::ALL());
            auto unit = Kokkos::subview(units, k, Kokkos::ALL(), Kokkos::ALL());

            // 1. We compute the pseudo inverse of A.
            ArborX::Details::symmetricPseudoInverseSVDKernel(A, diag, unit);
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        "matrix vector products",
        Kokkos::RangePolicy(execspace, 0, K),
        KOKKOS_LAMBDA(const size_t &k) {
            auto A = Kokkos::subview(As, k, Kokkos::ALL(), Kokkos::ALL());
            auto b = Kokkos::subview(bs, k, Kokkos::ALL());
            auto out = Kokkos::subview(coeffs, k, Kokkos::ALL());
            KokkosBlas::Experimental::serial_gemv('N', 1.0, A, b, 0.0, out);
        });
    Kokkos::fence();

    this->_coeffs =
        MatrixType(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                                      "this->_coeffs"),
                   K, N);
    Kokkos::deep_copy(this->_coeffs, coeffs);
    Kokkos::Profiling::popRegion(); // ! Prepare Interpolation
}

#endif /* ! CACHE_DATA_HPP */
